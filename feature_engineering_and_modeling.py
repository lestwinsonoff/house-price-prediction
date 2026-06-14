import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 全局中文字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

USE_PREPROCESSED_DATA = True

# 原始大文件路径
RAW_DATA_PATH = "../data/processed/hdb_data_with_bus_and_mall.csv"

# 预处理好的小文件路径
FINAL_DATA_PATH = "../data/processed/hdb_data_final_small.parquet"
# FINAL_DATA_PATH = "../data/processed/hdb_data_final_small.csv"

MODEL_SAVE_PATH = "../models/hdb_price_prediction_model.txt"
FEATURE_IMPORTANCE_PATH = "../reports/feature_importance.png"
FINAL_DATASET_FEATURES_PATH = "../reports/final_dataset_features.csv"
TRAINING_FEATURES_PATH = "../reports/training_features.csv"


def load_and_clean_data():
    if USE_PREPROCESSED_DATA:
        print("正在加载已经预处理好的最终数据集...")

        # 根据文件后缀选择读取方式
        if FINAL_DATA_PATH.endswith('.parquet'):
            df = pd.read_parquet(FINAL_DATA_PATH)
        else:
            df = pd.read_csv(FINAL_DATA_PATH, low_memory=False)

        print(f"加载成功！共 {len(df):,} 条记录，{len(df.columns)} 个特征")

        # 打印特征清单
        print("\n最终数据集特征清单：")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()

        if object_cols:
            print("\n文本类型特征：")
            for i, col in enumerate(object_cols, 1):
                print(f"   {i:2d}. {col}")

        print("\n数值类型特征：")
        for i, col in enumerate(numeric_cols, 1):
            print(f"   {i:2d}. {col} ({df[col].dtype})")

        return df

    print("正在加载原始大文件并进行预处理...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"原始数据：共 {len(df):,} 条记录，{len(df.columns)} 个特征")

    # 只保留训练和特征工程必须的17列
    REQUIRED_RAW_COLUMNS = [
        # 基础交易信息
        'year', 'quarter_num', 'resale_price',

        # 房屋核心属性
        'town', 'flat_type', 'flat_model', 'storey_range',
        'floor_area_sqm', 'lease_commence_date',

        # 配套设施数据
        'nearest_mrt_dist_m', 'nearest_bus_dist_m', 'nearest_mall_dist_m',
        'nearest_school_dist_m', 'nearest_park_dist_m', 'nearest_school_type',
        'crime_rate_per_1000',

        # app地图弹窗需要的展示字段
        'nearest_mrt_exit',

        # 地理坐标
        'latitude', 'longitude'
    ]

    # 立即过滤，只保留需要的列
    df = df[REQUIRED_RAW_COLUMNS].reset_index(drop=True)
    print(f"已删除所有无用列，剩余 {len(df.columns)} 个必要特征")

    # 数据类型压缩
    print("\n正在优化数据类型...")

    # 整数类型：从8字节→1-2字节
    df['year'] = df['year'].astype('int16')
    df['quarter_num'] = df['quarter_num'].astype('int8')
    df['lease_commence_date'] = df['lease_commence_date'].astype('int16')

    # 浮点类型：从8字节→4字节
    float_columns = [
        'floor_area_sqm', 'resale_price', 'crime_rate_per_1000',
        'nearest_mrt_dist_m', 'nearest_bus_dist_m', 'nearest_mall_dist_m',
        'nearest_school_dist_m', 'nearest_park_dist_m',
        'latitude', 'longitude'
    ]
    for col in float_columns:
        df[col] = df[col].astype('float32')

    memory_reduced = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"数据类型优化完成，内存占用减少 {memory_reduced:.1f}MB")

    # 保留原始2015年数据筛选逻辑
    print("\n正在筛选2015年以后的数据...")
    df = df[df['year'] >= 2015].reset_index(drop=True)
    print(f"   筛选后剩余 {len(df):,} 条记录")

    print("\n正在计算剩余租约...")
    if 'lease_commence_date' in df.columns:
        df['remaining_lease_years'] = 99 - (df['year'] - df['lease_commence_date'])
        df['remaining_lease_months'] = df['remaining_lease_years'] * 12
        if 'remaining_lease' in df.columns:
            df = df.drop(columns=['remaining_lease'])
        print(f"   成功计算剩余租约，缺失值已解决")

    print("\n正在对学校类型进行编码...")
    if 'nearest_school_type' in df.columns:
        school_type_mean = df.groupby('nearest_school_type')['resale_price'].mean()
        df['nearest_school_type_encoded'] = df['nearest_school_type'].map(school_type_mean)
        df = df.drop(columns=['nearest_school_type'])
        print(f"   学校类型编码完成")

    print("\n正在处理剩余的缺失值...")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        print(f"   发现 {len(missing_values)} 个特征有缺失值")
        for col, count in missing_values.items():
            print(f"     {col}: {count:,} 条 ({count / len(df) * 100:.2f}%)")
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("   没有发现缺失值")

    print("\n正在处理异常值...")
    df = df[(df['resale_price'] > 150000) & (df['resale_price'] < 1800000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 200)]
    df = df[(df['remaining_lease_years'] > 1) & (df['remaining_lease_years'] < 99)]
    df = df[(df['nearest_mall_dist_m'] > 0) & (df['nearest_mall_dist_m'] < 6000)]
    df = df[(df['nearest_mrt_dist_m'] > 0) & (df['nearest_mrt_dist_m'] < 4000)]
    print(f"   移除异常值后剩余 {len(df):,} 条记录")

    print("\n正在保存压缩后的最终数据集...")
    # 保存为CSV文件
    feature_df = pd.DataFrame({
        'feature_name': df.columns,
        'data_type': df.dtypes.astype(str),
        'is_numeric': df.dtypes.apply(lambda x: np.issubdtype(x, np.number)),
        'used_for_training': [col != 'nearest_mrt_exit' and col != 'latitude' and col != 'longitude' for col in
                              df.columns]
    })
    feature_df.to_csv(FINAL_DATASET_FEATURES_PATH, index=False)
    print(f"\n最终数据集特征清单已保存到：{FINAL_DATASET_FEATURES_PATH}")

    # CSV格式
    df.to_csv("../data/processed/hdb_data_final_small.csv", index=False)
    # Parquet格式
    df.to_parquet(
        "../data/processed/hdb_data_final_small.parquet",
        index=False,
        compression="snappy"
    )
    print("压缩后的数据集已保存！")
    print(f"   CSV格式：../data/processed/hdb_data_final_small.csv ")
    print(f"   Parquet格式：../data/processed/hdb_data_final_small.parquet ")

    return df


def feature_engineering(df):
    print("\n正在进行特征工程...")
    print("   正在对分类变量进行目标编码...")
    categorical_features = ['town', 'flat_type', 'flat_model', 'storey_range']

    # 保存原始市镇名称
    original_towns = df['town'].copy()

    for col in categorical_features:
        if col in df.columns:
            target_mean = df.groupby(col)['resale_price'].mean()
            df[f'{col}_encoded'] = df[col].map(target_mean)
            df = df.drop(columns=[col])

    # 添加成熟区/非成熟区核心特征
    print("   正在生成成熟区/非成熟区特征...")
    mature_towns = ['QUEENSTOWN', 'TOA PAYOH', 'ANG MO KIO', 'BUKIT TIMAH', 'CLEMENTI', 'BUKIT MERAH']
    df['is_mature_town'] = original_towns.isin(mature_towns).astype('int64')

    # 添加成熟区专属交互项
    print("   正在生成成熟区交互特征...")
    # 成熟区的大房子溢价更高（成熟区大户型稀缺）
    df['mature_area_interaction'] = df['is_mature_town'] * df['floor_area_sqm']
    # 成熟区的短租约贬值更快（成熟区租约是核心价值）
    df['mature_lease_interaction'] = df['is_mature_town'] * df['remaining_lease_years']

    print("   正在生成优化后的交通和商业特征...")
    df['mrt_within_200m'] = (df['nearest_mrt_dist_m'] <= 200).astype('int64')
    df['mrt_within_500m'] = (df['nearest_mrt_dist_m'] <= 500).astype('int64')
    df['mrt_within_1000m'] = (df['nearest_mrt_dist_m'] <= 1000).astype('int64')
    df['mrt_within_1500m'] = (df['nearest_mrt_dist_m'] <= 1500).astype('int64')

    df['mall_within_500m'] = (df['nearest_mall_dist_m'] <= 500).astype('int64')
    df['mall_within_1000m'] = (df['nearest_mall_dist_m'] <= 1000).astype('int64')
    df['mall_within_2000m'] = (df['nearest_mall_dist_m'] <= 2000).astype('int64')

    df['mrt_500m_area_interaction'] = df['mrt_within_500m'] * df['floor_area_sqm']
    df['mrt_1000m_area_interaction'] = df['mrt_within_1000m'] * df['floor_area_sqm']
    df['mall_500m_area_interaction'] = df['mall_within_500m'] * df['floor_area_sqm']
    df['mall_1000m_area_interaction'] = df['mall_within_1000m'] * df['floor_area_sqm']

    df['mrt_distance_log'] = np.log(df['nearest_mrt_dist_m'] + 1)
    df['bus_distance_log'] = np.log(df['nearest_bus_dist_m'] + 1)
    df['mall_distance_log'] = np.log(df['nearest_mall_dist_m'] + 1)

    df['mrt_distance_inv'] = 1 / (df['nearest_mrt_dist_m'] + 50)
    df['bus_distance_inv'] = 1 / (df['nearest_bus_dist_m'] + 30)
    df['school_distance_inv'] = 1 / (df['nearest_school_dist_m'] + 100)
    df['park_distance_inv'] = 1 / (df['nearest_park_dist_m'] + 100)
    df['mall_distance_inv'] = 1 / (df['nearest_mall_dist_m'] + 100)

    if 'nearest_mrt_dist_m' in df.columns and 'nearest_bus_dist_m' in df.columns:
        df['transport_score'] = 1 / (df['nearest_mrt_dist_m'] / 1000 + df['nearest_bus_dist_m'] / 500 + 0.1)
    if 'nearest_school_dist_m' in df.columns and 'nearest_park_dist_m' in df.columns:
        df['livability_score'] = 1 / (df['nearest_school_dist_m'] / 1000 + df['nearest_park_dist_m'] / 1000 + 0.1)
    if 'nearest_mall_dist_m' in df.columns:
        df['commercial_score'] = 1 / (df['nearest_mall_dist_m'] / 1000 + 0.1)

    print("   正在生成区域级聚合特征...")
    town_avg_lease = df.groupby('town_encoded')['remaining_lease_months'].transform('mean')
    df['town_avg_lease'] = town_avg_lease
    town_avg_area = df.groupby('town_encoded')['floor_area_sqm'].transform('mean')
    df['town_avg_area'] = town_avg_area
    df['town_avg_transport_score'] = df.groupby('town_encoded')['transport_score'].transform('mean')
    df['town_avg_commercial_score'] = df.groupby('town_encoded')['commercial_score'].transform('mean')

    print("   正在生成单个房源相对优势特征...")
    df['lease_diff_from_town_avg'] = df['remaining_lease_months'] - df['town_avg_lease']
    df['commercial_diff_from_town_avg'] = df['commercial_score'] - df['town_avg_commercial_score']

    print("\n正在验证所有特征都是数值类型...")
    # 支持所有数值类型
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if len(non_numeric_cols) > 0:
        print(f"   发现非数值类型特征：{non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)
        print(f"   已删除非数值类型特征")
    else:
        print("   所有特征都是数值类型")

    print("   正在生成扩展衍生特征...")
    df['remaining_lease_squared'] = df['remaining_lease_months'] ** 2
    df['quarter_since_2012'] = (df['year'] - 2012) * 4 + df['quarter_num']
    df['area_lease_interaction'] = df['floor_area_sqm'] * df['remaining_lease_years']
    df['transport_livability_interaction'] = df['transport_score'] * df['livability_score']
    df['area_transport_interaction'] = df['floor_area_sqm'] * df['transport_score']
    df['area_commercial_interaction'] = df['floor_area_sqm'] * df['commercial_score']
    df['transport_commercial_interaction'] = df['transport_score'] * df['commercial_score']

    print("   正在去除高相关性特征...")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    PROTECTED_FEATURES = [
        'nearest_mrt_dist_m', 'nearest_mall_dist_m',
        'mrt_within_500m', 'mall_within_500m',
        'transport_score', 'commercial_score',
        'area_commercial_interaction', 'transport_commercial_interaction',
        'remaining_lease_years',
        'mrt_500m_area_interaction', 'mrt_1000m_area_interaction',
        'mall_500m_area_interaction', 'mall_1000m_area_interaction',
        'is_mature_town',
        'mature_area_interaction',
        'mature_lease_interaction',
        'remaining_lease_months'
    ]
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > 0.98) and column not in PROTECTED_FEATURES:
            to_drop.append(column)

    if len(to_drop) > 0:
        df = df.drop(columns=to_drop)
        print(f"   去除了 {len(to_drop)} 个高相关性特征：{to_drop}")
    else:
        print("   未发现需要删除的高相关性特征")

    print("特征工程完成，共生成 {} 个特征".format(len(df.columns)))
    for i, col in enumerate(sorted(df.columns), 1):
        print(f"   {i:2d}. {col}")

    return df


def train_and_evaluate_model(df):
    print("\n正在训练LightGBM模型...")

    target = 'resale_price'
    exclude_features = [target]
    features = [col for col in df.columns if col not in exclude_features]

    print("模型训练实际使用的特征清单（共 {} 个）：".format(len(features)))
    for i, col in enumerate(sorted(features), 1):
        print(f"   {i:2d}. {col}")

    # 保存为CSV文件
    training_feature_df = pd.DataFrame({
        'feature_name': sorted(features),
        'feature_index': range(1, len(features) + 1)
    })
    training_feature_df.to_csv(TRAINING_FEATURES_PATH, index=False)
    print(f"\n训练特征清单已保存到：{TRAINING_FEATURES_PATH}")

    X = df[features]
    y = df[target]

    print("   正在按时间顺序划分训练集和测试集...")
    df_sorted = df.sort_values(['year', 'quarter_num'])
    test_years = [2025, 2026]
    train_df = df_sorted[df_sorted['year'] < 2025]
    test_df = df_sorted[df_sorted['year'].isin(test_years)]

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]

    print(f"   训练集：{len(X_train):,} 条记录（{train_df['year'].min()}-{train_df['year'].max()}年）")
    print(f"   测试集：{len(X_test):,} 条记录（{test_df['year'].min()}-{test_df['year'].max()}年）")
    print(f"   使用特征数量：{len(features)} 个")

    model = lgb.LGBMRegressor(
        n_estimators=1000,  # 最大迭代次数（最多训练1000棵决策树）
        learning_rate=0.02,  # 学习率，每棵树对最终预测结果的贡献权重
        max_depth=4,  # 单棵决策树的最大深度
        num_leaves=20,  # 单棵树最多叶子节点数
        min_child_samples=30,  # 每个叶子节点必须包含的最少样本数
        subsample=0.75,  # 行采样率，每次迭代随机选择75%的样本训练
        colsample_bytree=0.75,  # 列采样率，每次迭代随机选择75%的特征训练
        reg_alpha=6.0,  # L1正则化系数
        reg_lambda=7.0,  # L2正则化系数
        early_stopping_rounds=50,  # 早停机制，连续50轮无提升则停止
        random_state=42,  # 随机种子，保证实验可复现
        verbosity=-1,  # 关闭所有训练日志输出
        importance_type='gain'  # 按信息增益计算特征重要性
    )

    print("\n   开始训练模型，启用早停机制...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse',
        callbacks=[lgb.log_evaluation(period=10)]
    )

    print(f"\n   最佳迭代次数：{model.best_iteration_}")

    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration_)
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration_)

    print("\n模型评估结果：")
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # 用测试集数据计算，最准确反映模型实际预测能力
    test_area = X_test['floor_area_sqm']
    true_price_per_sqm = y_test / test_area
    pred_price_per_sqm = y_pred_test / test_area
    sqm_absolute_error = np.abs(true_price_per_sqm - pred_price_per_sqm)

    # 每平方米误差指标
    test_mae_per_sqm = sqm_absolute_error.mean()
    test_median_mae_per_sqm = sqm_absolute_error.median()
    test_rmse_per_sqm = np.sqrt(np.mean(sqm_absolute_error ** 2))

    print(f"   训练集R²得分：{train_r2:.4f}")
    print(f"   测试集R²得分：{test_r2:.4f}")
    print(f"\n   总价维度误差")
    print(f"   测试集平均绝对误差：${test_mae:,.2f}")
    print(f"   测试集均方根误差：${test_rmse:,.2f}")
    print(f"\n   每平方米维度误差")
    print(f"   测试集每平方米平均误差：${test_mae_per_sqm:,.2f}/㎡")
    print(f"   测试集每平方米中位数误差：${test_median_mae_per_sqm:,.2f}/㎡")
    print(f"   测试集每平方米均方根误差：${test_rmse_per_sqm:,.2f}/㎡")

    absolute_error = np.abs(y_test - y_pred_test)
    mape = np.mean(absolute_error / y_test) * 100
    accuracy_10pct = np.mean(absolute_error / y_test < 0.1) * 100
    accuracy_5pct = np.mean(absolute_error / y_test < 0.05) * 100
    print(f"\n   平均绝对百分比误差：{mape:.2f}%")
    print(f"   预测误差在5%以内的比例：{accuracy_5pct:.2f}%")
    print(f"   预测误差在10%以内的比例：{accuracy_10pct:.2f}%")

    print("\n正在进行时间序列交叉验证...")
    tscv = TimeSeriesSplit(
        n_splits=5,
        max_train_size=240000,
        gap=0
    )

    cv_r2_scores = []
    cv_mae_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model_cv = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=20,
            min_child_samples=30,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=6.0,
            reg_lambda=7.0,
            early_stopping_rounds=50,
            random_state=42,
            verbosity=-1,
            importance_type='gain'
        )
        model_cv.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_test_cv, y_test_cv)],
            eval_metric='rmse',
            callbacks=[lgb.log_evaluation(period=False)]
        )

        y_pred_cv = model_cv.predict(X_test_cv, num_iteration=model_cv.best_iteration_)
        cv_r2 = r2_score(y_test_cv, y_pred_cv)
        cv_mae = mean_absolute_error(y_test_cv, y_pred_cv)

        cv_r2_scores.append(cv_r2)
        cv_mae_scores.append(cv_mae)
        print(f"   折 {fold + 1}: R²={cv_r2:.4f}, MAE=${cv_mae:,.2f}")

    print(f"\n   交叉验证平均R²得分：{np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")
    print(f"   交叉验证平均MAE：${np.mean(cv_mae_scores):,.2f}")

    print("\n特征重要性排名（前25）：")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    mall_features = [
        'commercial_score', 'mall_within_500m', 'mall_within_1000m',
        'mall_distance_log', 'mall_distance_inv', 'town_avg_commercial_score',
        'area_commercial_interaction', 'transport_commercial_interaction'
    ]

    mrt_features = [
        'nearest_mrt_dist_m', 'mrt_within_200m', 'mrt_within_500m', 'mrt_within_1000m',
        'mrt_distance_log', 'mrt_distance_inv', 'transport_score',
        'mrt_500m_area_interaction', 'mrt_1000m_area_interaction', 'town_avg_transport_score'
    ]

    for i, row in feature_importance.head(25).iterrows():
        if row['feature'] in mall_features:
            print(f"   {i + 1}. {row['feature']} (商业) - {row['importance']:.4f}")
        elif row['feature'] in mrt_features:
            print(f"   {i + 1}. {row['feature']} (地铁) - {row['importance']:.4f}")
        else:
            print(f"   {i + 1}. {row['feature']} - {row['importance']:.4f}")

    feature_importance.to_csv("../reports/feature_importance_full.csv", index=False)
    print(f"\n完整特征重要性已保存到：../reports/feature_importance_full.csv")

    plt.figure(figsize=(16, 12))
    top_features = feature_importance.head(25).copy()

    def get_feature_type(feature):
        if feature in mall_features:
            return '商业配套'
        elif feature in mrt_features:
            return '地铁交通'
        else:
            return '其他'

    top_features['feature_type'] = top_features['feature'].apply(get_feature_type)

    sns.barplot(
        x='importance',
        y='feature',
        hue='feature_type',
        data=top_features,
        palette={'商业配套': '#e74c3c', '地铁交通': '#3498db', '其他': '#95a5a6'},
        legend=True
    )

    plt.legend(title='特征类型', loc='upper right')
    plt.title('Top 25 Feature Importance (Red=Commercial, Blue=MRT)', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300)
    print(f"特征重要性图表已保存到：{FEATURE_IMPORTANCE_PATH}")

    model.booster_.save_model(MODEL_SAVE_PATH)
    print(f"训练好的模型已保存到：{MODEL_SAVE_PATH}")

    return model, feature_importance


if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../reports", exist_ok=True)

    print("新加坡HDB房价预测模型训练（LightGBM）")

    df = load_and_clean_data()
    df = feature_engineering(df)
    model, feature_importance = train_and_evaluate_model(df)
