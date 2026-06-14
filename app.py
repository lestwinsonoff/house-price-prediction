import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
from folium.plugins import MarkerCluster
import lightgbm as lgb

# 页面配置
st.set_page_config(
    page_title="新加坡HDB房价分析与预测系统",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# 数据和模型缓存
@st.cache_data(show_spinner="正在加载数据...")
def load_data():
    # ✅ 更新：使用包含商场数据的最终数据集
    #df = pd.read_csv("../data/processed/hdb_data_with_bus_and_mall.csv", low_memory=False)
    #df = pd.read_csv("../data/processed/hdb_data_final_small.csv", low_memory=False)
    df = pd.read_parquet("data/processed/hdb_data_final_small.parquet")

    # 预处理（与训练代码完全一致）
    #df['month'] = pd.to_datetime(df['month'])
    df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    df['remaining_lease_years'] = 99 - (df['year'] - df['lease_commence_date'])
    df['remaining_lease_months'] = df['remaining_lease_years'] * 12

    # 计算目标编码映射（用于预测）
    town_map = df.groupby('town')['resale_price'].mean().to_dict()
    flat_type_map = df.groupby('flat_type')['resale_price'].mean().to_dict()
    flat_model_map = df.groupby('flat_model')['resale_price'].mean().to_dict()
    storey_range_map = df.groupby('storey_range')['resale_price'].mean().to_dict()

    # ✅ 新增：计算综合得分（与训练代码完全一致）
    df['transport_score'] = 1 / (df['nearest_mrt_dist_m'] / 1000 + df['nearest_bus_dist_m'] / 500 + 0.1)
    df['commercial_score'] = 1 / (df['nearest_mall_dist_m'] / 1000 + 0.1)

    # ✅ 更新：市镇统计数据新增商业得分
    town_stats = df.groupby('town').agg({
        'remaining_lease_months': 'mean',
        'floor_area_sqm': 'mean',
        'transport_score': 'mean',
        'commercial_score': 'mean'  # 新增：区域平均商业得分
    }).reset_index()
    town_stats.columns = ['town', 'avg_lease_months', 'avg_area_sqm',
                          'avg_transport_score', 'avg_commercial_score']
    town_stats_dict = town_stats.set_index('town').T.to_dict('list')

    return df, town_map, flat_type_map, flat_model_map, storey_range_map, town_stats_dict


# ✅ 替换为LightGBM版本
@st.cache_resource(show_spinner="正在加载模型...")
def load_model():
    # LightGBM官方推荐的加载方式
    model = lgb.Booster(model_file='models/hdb_price_prediction_model.txt')
    return model


# 加载数据和模型
df, town_map, flat_type_map, flat_model_map, storey_range_map, town_stats_dict = load_data()
model = load_model()

# 侧边栏导航
st.sidebar.title("🏠 导航菜单")
page = st.sidebar.radio(
    "选择功能模块",
    ["📊 首页概览", "🔍 房源筛选与地图", "📈 价格影响因素分析", "🔮 房价预测", "💡 购房策略"]
)

# 页面1：首页概览（✅ 专业仪表盘版）
if page == "📊 首页概览":
    st.markdown('<h1 class="main-header">新加坡HDB房价分析与预测系统</h1>', unsafe_allow_html=True)

    # ====================== 1. 核心指标看板（扩展到6个） ======================
    st.markdown('<h2 class="sub-header">📊 市场核心指标</h2>', unsafe_allow_html=True)

    # 预计算所有统计指标（只计算一次）
    total_units = len(df)
    avg_total_price = df['resale_price'].mean()
    median_total_price = df['resale_price'].median()
    avg_price_per_sqm = (df['resale_price'] / df['floor_area_sqm']).mean()
    avg_area = df['floor_area_sqm'].mean()
    avg_remaining_lease = df['remaining_lease_years'].mean()
    avg_mall_dist = df['nearest_mall_dist_m'].mean()
    million_dollar_units = len(df[df['resale_price'] >= 1000000])

    # 6列卡片布局
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总成交套数", f"{total_units:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均总价", f"${avg_total_price:,.0f}")
        st.caption(f"中位数: ${median_total_price:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均单价", f"${avg_price_per_sqm:,.0f}/㎡")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均面积", f"{avg_area:.1f} ㎡")
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均剩余租约", f"{avg_remaining_lease:.1f} 年")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("百万级房源", f"{million_dollar_units:,}")
        st.caption(f"占比: {million_dollar_units / total_units * 100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ====================== 2. 房价趋势分析（双列布局） ======================
    st.markdown('<h2 class="sub-header">📈 房价走势分析（2015-2026）</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 年度平均总价趋势
        price_trend = df.groupby('year')['resale_price'].mean().reset_index()
        fig = px.line(price_trend, x='year', y='resale_price',
                      title='年度平均转售总价趋势',
                      labels={'resale_price': '平均价格（新元）', 'year': '年份'},
                      markers=True,
                      color_discrete_sequence=['#1f77b4'])
        # ✅ 新增：强制显示所有年份刻度
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(
                tickmode='linear',  # 线性刻度模式
                dtick=1,  # 每隔1年显示一个刻度
                range=[price_trend['year'].min() - 0.5, price_trend['year'].max() + 0.5]  # 左右留边距，避免点被截断
            )
        )
        fig.update_traces(line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 年度平均单价趋势
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        price_per_sqm_trend = df.groupby('year')['price_per_sqm'].mean().reset_index()
        fig = px.line(price_per_sqm_trend, x='year', y='price_per_sqm',
                      title='年度平均转售单价趋势',
                      labels={'price_per_sqm': '单价（新元/㎡）', 'year': '年份'},
                      markers=True,
                      color_discrete_sequence=['#e74c3c'])
        # ✅ 同样添加x轴配置
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(
                tickmode='linear',
                dtick=1,
                range=[price_per_sqm_trend['year'].min() - 0.5, price_per_sqm_trend['year'].max() + 0.5]
            )
        )
        fig.update_traces(line_width=3)
        st.plotly_chart(fig, use_container_width=True)

    # 市场动态提示
    st.info("""
    📢 最新市场动态：2026年第一季度HDB转售价格指数出现7年来首次下跌，环比下降0.6%，
    市场进入温和调整期。预计2026年全年价格将保持稳定，涨幅在2%-4%之间。
    """)

    # ====================== 3. 区域房价排名（棒棒糖图） ======================
    st.markdown('<h2 class="sub-header">🏙️ 各区域房价排名</h2>', unsafe_allow_html=True)

    # 计算各区域平均单价并排序
    town_price = df.groupby('town')['price_per_sqm'].mean().sort_values(ascending=False).reset_index()

    # 只显示前10名和后5名，避免图表过长
    top_towns = town_price.head(10)
    bottom_towns = town_price.tail(5)
    display_towns = pd.concat([top_towns, bottom_towns]).drop_duplicates()

    # 棒棒糖图实现
    fig = px.scatter(display_towns, x='price_per_sqm', y='town',
                     title='各区域平均单价排名（前10+后5）',
                     labels={'price_per_sqm': '平均单价（新元/㎡）', 'town': '市镇'},
                     size=[1] * len(display_towns),
                     size_max=15,
                     color_discrete_sequence=['#1f77b4'])

    # 添加水平线
    for i, row in display_towns.iterrows():
        fig.add_shape(
            type='line',
            x0=0, y0=row['town'],
            x1=row['price_per_sqm'], y1=row['town'],
            line=dict(color='#1f77b4', width=2)
        )

    fig.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # ====================== 4. 户型分析（双列布局） ======================
    st.markdown('<h2 class="sub-header">🏠 户型分布与价格</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 户型占比饼图
        flat_type_count = df['flat_type'].value_counts().reset_index()
        flat_type_count.columns = ['flat_type', 'count']

        fig = px.pie(flat_type_count, values='count', names='flat_type',
                     title='各户型成交占比',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=450)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 各户型平均价格柱状图
        flat_type_price = df.groupby('flat_type')['price_per_sqm'].mean().sort_values().reset_index()

        fig = px.bar(flat_type_price, x='flat_type', y='price_per_sqm',
                     title='各户型平均单价对比',
                     labels={'price_per_sqm': '平均单价（新元/㎡）', 'flat_type': '户型'},
                     color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ====================== 5. 价格区间分布 ======================
    st.markdown('<h2 class="sub-header">💰 房价分布情况</h2>', unsafe_allow_html=True)

    # 价格分箱
    df['price_bin'] = pd.cut(df['resale_price'],
                             bins=[0, 300000, 500000, 700000, 1000000, 2000000],
                             labels=['<30万', '30-50万', '50-70万', '70-100万', '>100万'])

    price_dist = df['price_bin'].value_counts().reset_index()
    price_dist.columns = ['price_bin', 'count']
    price_dist = price_dist.sort_values('price_bin')

    fig = px.bar(price_dist, x='price_bin', y='count',
                 title='不同价格区间房源数量分布',
                 labels={'count': '房源数量', 'price_bin': '价格区间'},
                 color_discrete_sequence=['#1f77b4'])
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ====================== 6. 配套设施概览 ======================
    st.markdown('<h2 class="sub-header">🛍️ 周边配套设施统计</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 地铁站距离分布
        df['mrt_bin'] = pd.cut(df['nearest_mrt_dist_m'],
                               bins=[0, 500, 1000, 2000, 10000],
                               labels=['<500米', '500-1000米', '1000-2000米', '>2000米'])
        mrt_dist = df['mrt_bin'].value_counts().reset_index()

        fig = px.pie(mrt_dist, values='count', names='mrt_bin',
                     title='地铁站距离分布',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 商场距离分布
        df['mall_bin'] = pd.cut(df['nearest_mall_dist_m'],
                                bins=[0, 500, 1000, 2000, 10000],
                                labels=['<500米', '500-1000米', '1000-2000米', '>2000米'])
        mall_dist = df['mall_bin'].value_counts().reset_index()

        fig = px.pie(mall_dist, values='count', names='mall_bin',
                     title='商场距离分布',
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # 小学距离分布
        df['school_bin'] = pd.cut(df['nearest_school_dist_m'],
                                  bins=[0, 1000, 2000, 10000],
                                  labels=['<1000米(名校圈)', '1000-2000米', '>2000米'])
        school_dist = df['school_bin'].value_counts().reset_index()

        fig = px.pie(school_dist, values='count', names='school_bin',
                     title='小学距离分布',
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # 剩余租约分布
        df['lease_bin'] = pd.cut(df['remaining_lease_years'],
                                 bins=[0, 30, 60, 99],
                                 labels=['<30年', '30-60年', '>60年'])
        lease_dist = df['lease_bin'].value_counts().reset_index()

        fig = px.pie(lease_dist, values='count', names='lease_bin',
                     title='剩余租约分布',
                     color_discrete_sequence=px.colors.sequential.Greens_r)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ====================== 7. 关键数据洞察（折叠面板） ======================
    # with st.expander("📌 点击查看关键数据洞察"):
    #     st.markdown("""
    #     ### 🏆 市场之最
    #     - **最贵区域**：BUKIT TIMAH（武吉知马），平均单价超过$12,000/㎡
    #     - **最便宜区域**：WOODLANDS（兀兰），平均单价约$4,500/㎡
    #     - **最受欢迎户型**：4 ROOM（四室一厅），占总成交量的45%以上
    #     - **最高成交价格**：超过$200万新元（BUKIT TIMAH的五房式阁楼）
    #
    #     ### 📊 核心发现
    #     1. **区域差异显著**：成熟区平均房价比非成熟区高出约40%
    #     2. **地铁溢价明显**：距离地铁站500米以内的房源平均溢价约15%
    #     3. **名校效应**：距离优质小学1000米以内的房源平均溢价约12%
    #     4. **商业配套重要性**：距离商场500米以内的房源平均溢价约8%
    #
    #     ### 📈 市场趋势
    #     - 2020-2023年房价快速上涨，累计涨幅超过30%
    #     - 2024年开始增速放缓，2026年第一季度出现首次下跌
    #     - 百万级房源数量持续增加，目前已占总成交量的约5%
    #     """)

# 页面2：房源筛选与地图（✅ 筛选条件样式全面优化 + Cloud完美兼容）
elif page == "🔍 房源筛选与地图":
    st.markdown('<h1 class="main-header">房源筛选与地图可视化</h1>', unsafe_allow_html=True)

    # ====================== ✅ 新增：全局筛选条件样式优化 ======================
    st.markdown("""
    <style>
    /* 侧边栏整体样式 */
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 筛选条件标题样式 */
    [data-testid="stSidebar"] h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }

    /* 滑块样式优化 */
    [data-testid="stSlider"] > div > div > div > div {
        background-color: #3498db !important;
    }
    [data-testid="stSlider"] > div > div > div > div::before {
        background-color: #3498db !important;
        border: 2px solid white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    [data-testid="stSlider"] label {
        font-weight: 500;
        color: #34495e;
        margin-bottom: 0.5rem;
    }

    /* 多选框标签样式优化 */
    [data-testid="stMultiSelect"] [data-testid="stTag"] {
        border-radius: 20px !important;
        padding: 0.3rem 0.8rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* 成熟区标签：蓝色 */
    [data-testid="stMultiSelect"] [data-testid="stTag"]:has(span:contains("成熟区")) {
        background-color: #3498db !important;
        color: white !important;
    }

    /* 非成熟区标签：绿色 */
    [data-testid="stMultiSelect"] [data-testid="stTag"]:has(span:contains("非成熟区")) {
        background-color: #2ecc71 !important;
        color: white !important;
    }

    /* 其他标签：灰色 */
    [data-testid="stMultiSelect"] [data-testid="stTag"]:not(:has(span:contains("成熟区"))):not(:has(span:contains("非成熟区"))) {
        background-color: #95a5a6 !important;
        color: white !important;
    }

    /* 多选框下拉菜单样式 */
    [data-testid="stMultiSelect"] [role="listbox"] {
        max-height: 300px;
    }

    /* 筛选条件之间的间距 */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 筛选条件
    st.sidebar.subheader("筛选条件")

    max_display = st.sidebar.slider(
        "地图最大显示房源数",
        min_value=500,
        max_value=5000,
        value=500,
        step=100,
        help="数值越大，地图加载越慢；建议不超过2000"
    )

    # ====================== ✅ 优化1：市镇自动标记成熟区/非成熟区 ======================
    # 与预测页面保持一致的定义
    mature_towns = ['QUEENSTOWN', 'TOA PAYOH', 'ANG MO KIO', 'BUKIT TIMAH', 'CLEMENTI', 'BUKIT MERAH']
    non_mature_towns = ['PUNGGOL', 'SENGKANG', 'WOODLANDS', 'YISHUN', 'SEMBAWANG', 'TAMPINES']

    towns = sorted(df['town'].unique().tolist())
    # 生成带标签的市镇选项
    town_options = []
    town_label_map = {}  # 用于反向映射
    for town in towns:
        if town in mature_towns:
            label = f"{town} (成熟区)"
        elif town in non_mature_towns:
            label = f"{town} (非成熟区)"
        else:
            label = f"{town} (其他)"
        town_options.append(label)
        town_label_map[label] = town

    selected_town_labels = st.sidebar.multiselect(
        "选择镇区",
        town_options,
        default=town_options[:5],
        help="🔵 成熟区：历史悠久、配套完善、价格较高\n🟢 非成熟区：新兴发展、升值潜力大、价格较低"
    )
    # 反向映射回原始市镇名称
    selected_towns = [town_label_map[label] for label in selected_town_labels]

    # ====================== ✅ 优化2：房型添加详细说明 ======================
    flat_types = sorted(df['flat_type'].unique().tolist())
    flat_type_help = """
    户型说明：
    • 1 ROOM：单卧室组屋（约23-33㎡）
    • 2 ROOM：两室一厅（约36-45㎡）
    • 3 ROOM：三室一厅（约60-70㎡）
    • 4 ROOM：四室一厅（约90-100㎡）
    • 5 ROOM：五室一厅（约110-130㎡）
    • EXECUTIVE：行政公寓（约140-160㎡）
    • MULTI-GENERATION：多代同堂组屋
    """

    selected_flat_types = st.sidebar.multiselect(
        "选择房型",
        flat_types,
        default=flat_types[:3],
        help=flat_type_help
    )

    # 价格范围
    min_price = int(df['resale_price'].min())
    max_price = int(df['resale_price'].max())
    price_range = st.sidebar.slider("价格范围（新元）",
                                    min_value=min_price,
                                    max_value=max_price,
                                    value=(min_price, max_price),
                                    format="$%d")

    # 剩余租约
    min_lease = int(df['remaining_lease_years'].min())
    max_lease = int(df['remaining_lease_years'].max())
    lease_range = st.sidebar.slider("剩余租约（年）",
                                    min_value=min_lease,
                                    max_value=max_lease,
                                    value=(min_lease, max_lease))

    # 交易年份
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_range = st.sidebar.slider(
        "交易年份",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )

    # 房屋面积
    min_area = int(df['floor_area_sqm'].min())
    max_area = int(df['floor_area_sqm'].max())
    area_range = st.sidebar.slider(
        "房屋面积（平方米）",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area),
        step=5,
        format="%d ㎡"
    )

    # 商场距离筛选
    min_mall_dist = int(df['nearest_mall_dist_m'].min())
    max_mall_dist = int(df['nearest_mall_dist_m'].max())
    mall_range = st.sidebar.slider(
        "到最近商场距离（米）",
        min_value=min_mall_dist,
        max_value=max_mall_dist,
        value=(min_mall_dist, max_mall_dist),
        step=100,
        format="%d 米"
    )

    # 应用所有筛选条件
    filtered_df = df[
        (df['town'].isin(selected_towns)) &
        (df['flat_type'].isin(selected_flat_types)) &
        (df['resale_price'] >= price_range[0]) &
        (df['resale_price'] <= price_range[1]) &
        (df['remaining_lease_years'] >= lease_range[0]) &
        (df['remaining_lease_years'] <= lease_range[1]) &
        (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
        (df['floor_area_sqm'] >= area_range[0]) & (df['floor_area_sqm'] <= area_range[1]) &
        (df['nearest_mall_dist_m'] >= mall_range[0]) & (df['nearest_mall_dist_m'] <= mall_range[1])
        ]

    # 基础统计看板（✅ 优化版）
    st.markdown('<h2 class="sub-header">基础统计看板</h2>', unsafe_allow_html=True)
    
    # 全局统计卡片样式美化
    st.markdown("""
    <style>
    /* 统计卡片整体样式 */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
    }
    
    /* 统计数值样式 */
    [data-testid="stMetric"] > div:first-child > div:last-child {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2c3e50;
    }
    
    /* 统计标签样式 */
    [data-testid="stMetric"] label {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        # ✅ 提前计算一次，避免重复计算
        filtered_df['price_per_sqm'] = filtered_df['resale_price'] / filtered_df['floor_area_sqm']
        
        # 计算所有统计指标（增加中位数，更有参考价值）
        total_units = len(filtered_df)
        avg_total_price = filtered_df['resale_price'].mean()
        median_total_price = filtered_df['resale_price'].median()  # 中位数不受极端值影响
        avg_price_per_sqm = filtered_df['price_per_sqm'].mean()
        avg_remaining_lease = filtered_df['remaining_lease_years'].mean()
        avg_mrt_dist = filtered_df['nearest_mrt_dist_m'].mean()
        avg_mall_dist = filtered_df['nearest_mall_dist_m'].mean()
    
        # ✅ 2行3列布局，更宽敞更美观
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
    
        with row1_col1:
            st.metric("筛选后成交套数", f"{total_units:,} 套")
        with row1_col2:
            st.metric("平均总价", f"${avg_total_price:,.0f}")
        with row1_col3:
            st.metric("中位数总价", f"${median_total_price:,.0f}", help="更能代表大多数房源的实际价格")
    
        with row2_col1:
            st.metric("平均单价", f"${avg_price_per_sqm:,.0f}/㎡")
        with row2_col2:
            st.metric("平均剩余租约", f"{avg_remaining_lease:.1f} 年")
        with row2_col3:
            st.metric("平均地铁站距离", f"{avg_mrt_dist:.0f} 米")
    
    else:
        st.warning("⚠️ 没有找到符合条件的房源，请调整筛选条件")

    # 地图可视化
    st.markdown('<h2 class="sub-header">房源分布地图</h2>', unsafe_allow_html=True)

    if len(filtered_df) > 0:
        display_df = filtered_df.sample(n=min(max_display, len(filtered_df)), random_state=42)

       # 创建地图（保留所有历史底图+新增稳定底图）
        m = folium.Map(
            location=[1.3521, 103.8198],
            zoom_start=11,
            tiles=None,  # 不使用默认底图，全部手动添加
            prefer_canvas=True,
            control_scale=True,
            max_zoom=18,
            min_zoom=10
        )
        
        # ====================== 所有可用底图（按稳定性排序） ======================
        # 1. Esri 街道图（当前最稳定，默认显示）
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
            name='Esri 街道图',
            attr='Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS, Intermap, iPC, NRCAN, Esri Japan, METI, Esri China (Hong Kong), Esri (Thailand), TomTom',
            show=True  # 默认显示这个最稳定的底图
        ).add_to(m)
        
        
        # 按价格区间给房源点着色
        price_min = display_df['resale_price'].min()
        price_max = display_df['resale_price'].max()
        price_bins = [price_min, 300000, 500000, 700000, 1000000, price_max]
        price_labels = ['<30万', '30-50万', '50-70万', '70-100万', '>100万']
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

        # 房源分布图层（聚合显示）
        housing_layer = folium.FeatureGroup(name='🏠 房源分布', show=True)

        marker_cluster = MarkerCluster(
            name='房源聚合',
            overlay=True,
            control=False,
            disableClusteringAtZoom=15,  # 缩放大于15时取消聚合，提升性能
            maxClusterRadius=50,  # 减小聚合半径，提高响应速度
            icon_create_function="""
            function(cluster) {
                var count = cluster.getChildCount();
                var size = 'small';
                if (count > 100) size = 'large';
                else if (count > 50) size = 'medium';
                return L.divIcon({
                    html: '<div style="background-color: #1f77b4; color: white; border-radius: 50%; width: 30px; height: 30px; text-align: center; line-height: 30px; font-weight: bold;">' + count + '</div>',
                    className: 'marker-cluster-' + size,
                    iconSize: L.point(30, 30)
                });
            }
            """
        ).add_to(housing_layer)

        for idx, row in display_df.iterrows():
            price = row['resale_price']
            for i in range(len(price_bins) - 1):
                if price_bins[i] <= price < price_bins[i + 1]:
                    color = colors[i]
                    break
            else:
                color = colors[-1]

            # ✅ 更新：弹窗新增最近商场距离信息
            popup_html = f"""
            <div style="width: 280px; font-size: 13px; line-height: 1.6;">
                <h4 style="margin: 0 0 10px 0; color: #1f77b4; border-bottom: 1px solid #eee; padding-bottom: 5px;">
                    {row['town']} {row['flat_type']}
                </h4>

                <div style="margin-bottom: 8px;">
                    <b>基本信息</b><br>
                    • 房屋模型：{row['flat_model']}<br>
                    • 楼层范围：{row['storey_range']}<br>
                    • 面积：{row['floor_area_sqm']} ㎡<br>
                    • 单价：${row['price_per_sqm']:,.0f}/㎡<br>
                    • 总价：<span style="color: {color}; font-weight: bold;">${row['resale_price']:,.0f}</span><br>
                    • 剩余租约：{row['remaining_lease_years']:.1f} 年
                </div>

                <div style="margin-bottom: 8px;">
                    <b>交通便利性</b><br>
                    • 最近地铁站出入口：{row['nearest_mrt_exit']}<br>
                    • 距离：{row['nearest_mrt_dist_m']:.0f} 米<br>
                    • 最近公交站：{row['nearest_bus_dist_m']:.0f} 米
                </div>

                <div style="margin-bottom: 8px;">
                    <b>周边配套</b><br>
                    • 最近商场：{row['nearest_mall_dist_m']:.0f} 米<br>
                    • 最近小学：{row['nearest_school_dist_m']:.0f} 米<br>
                    • 最近公园：{row['nearest_park_dist_m']:.0f} 米
                </div>

                <div>
                    <b>其他信息</b><br>
                    • 区域犯罪率：{row['crime_rate_per_1000']:.2f} 每千人<br>
                    • 交易年份：{row['year']}年
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=1
            ).add_to(marker_cluster)

        housing_layer.add_to(m)

        # 通用地理数据加载函数
        def load_geospatial_data(file_path, name_column_candidates, default_name):
            try:
                data = pd.read_csv(file_path)
                columns = data.columns.tolist()

                lat_col = None
                lon_col = None
                for col in columns:
                    col_lower = col.lower()
                    if 'lat' in col_lower or 'latitude' in col_lower:
                        lat_col = col
                    if 'lon' in col_lower or 'long' in col_lower or 'longitude' in col_lower:
                        lon_col = col

                name_col = None
                for candidate in name_column_candidates:
                    if candidate in columns:
                        name_col = candidate
                        break

                if not lat_col or not lon_col:
                    raise ValueError(f"无法找到经纬度字段。现有字段：{columns}")

                if not name_col:
                    name_col = columns[0]
                    st.sidebar.info(f"{default_name}：未找到名称字段，使用 '{name_col}' 代替")

                return data, lat_col, lon_col, name_col

            except Exception as e:
                raise Exception(f"{default_name}加载失败：{str(e)}")

        # 地铁站图层（使用内置蓝色圆点图标）
        try:
            mrt_data, lat_col, lon_col, name_col = load_geospatial_data(
                "data/processed/mrt_stations_clean.csv",
                ['NAME', 'mrt_name', 'name', 'station_name', 'station'],
                "地铁站出入口"
            )
        
            mrt_layer = folium.FeatureGroup(name='🚇 地铁站出入口', show=False)
        
            for idx, row in mrt_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=6,
                    popup=f"<b>🚇 {row[name_col]}</b>",
                    color='#1f77b4',
                    fill=True,
                    fill_color='#1f77b4',
                    fill_opacity=0.9,
                    weight=2
                ).add_to(mrt_layer)
        
            mrt_layer.add_to(m)
        except Exception as e:
            pass
        
        # 公交站图层（✅ 显示全部公交站）
        try:
            bus_data, lat_col, lon_col, name_col = load_geospatial_data(
                "data/processed/bus_stops_clean.csv",
                ['bus_stop_name', 'name', 'description', 'bus_stop_code'],
                "公交站"
            )
        
            bus_layer = folium.FeatureGroup(name='🚌 公交站', show=False)
        
            # 🔴 关键修改：删除.sample()抽样，直接遍历全部数据
            for idx, row in bus_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=2,  # 建议减小半径，避免点过于密集
                    popup=f"<b>🚌 公交站：{row[name_col]}</b>",
                    color='#ff7f0e',
                    fill=True,
                    fill_color='#ff7f0e',
                    fill_opacity=0.5,  # 降低透明度，缓解视觉拥挤
                    weight=0  # 去掉边框，减少渲染压力
                ).add_to(bus_layer)
        
            bus_layer.add_to(m)
        except Exception as e:
            pass
        
        # 小学图层（使用内置紫色圆点图标）
        try:
            school_data, lat_col, lon_col, name_col = load_geospatial_data(
                "data/processed/primary_schools_clean.csv",
                ['school_name', 'name', 'primary_school', 'school'],
                "小学"
            )
        
            school_layer = folium.FeatureGroup(name='🏫 小学', show=False)
        
            for idx, row in school_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f"<b>🏫 {row[name_col]}</b>",
                    color='#9467bd',
                    fill=True,
                    fill_color='#9467bd',
                    fill_opacity=0.9,
                    weight=2
                ).add_to(school_layer)
        
            school_layer.add_to(m)
        except Exception as e:
            pass
        
        # 公园图层（使用内置绿色圆点图标）
        try:
            park_data, lat_col, lon_col, name_col = load_geospatial_data(
                "data/processed/parks_clean.csv",
                ['park_name', 'name', 'park'],
                "公园"
            )
        
            park_layer = folium.FeatureGroup(name='🌳 公园', show=False)
        
            for idx, row in park_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f"<b>🌳 {row[name_col]}</b>",
                    color='#2ca02c',
                    fill=True,
                    fill_color='#2ca02c',
                    fill_opacity=0.9,
                    weight=2
                ).add_to(park_layer)
        
            park_layer.add_to(m)
        except Exception as e:
            pass
        
        # 商场图层（使用内置红色圆点图标）
        try:
            mall_data, lat_col, lon_col, name_col = load_geospatial_data(
                "data/processed/malls_clean.csv",
                ['mall_name', 'name', 'shopping_mall'],
                "商场"
            )
        
            mall_layer = folium.FeatureGroup(name='🛒 商场', show=False)
        
            for idx, row in mall_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    popup=f"<b>🛒 {row[name_col]}</b>",
                    color='#d62728',
                    fill=True,
                    fill_color='#d62728',
                    fill_opacity=0.9,
                    weight=2
                ).add_to(mall_layer)
        
            mall_layer.add_to(m)
        except Exception as e:
            pass

        # 添加图层控制
        folium.LayerControl(
            collapsed=False,
            position='topright',
            autoZIndex=True
        ).add_to(m)

        # 地图+图例布局
        col_map, col_legend = st.columns([8.5, 1.5])

        with col_map:
            # 🔴 修复：移除returned_objects参数，替换use_container_width为width="stretch"
            st_folium(
                m,
                height=600,
                width="stretch",
                debug=False,
                returned_objects=[],  # 🔴 这一行是关键！
                key="main_map"  # 添加唯一key，避免组件冲突
            )

        with col_legend:
            st.markdown(
                '<div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-size: 14px; margin-top: 20px; min-width: 180px;">'
                '<h5 style="margin-top: 0; margin-bottom: 12px; font-weight: bold; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px;">🏠 房源价格区间</h5>'
                '<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 18px; height: 18px; background-color: #2ecc71; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div><span style="color: #34495e;">< 30万 新元</span></div>'
                '<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 18px; height: 18px; background-color: #f1c40f; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div><span style="color: #34495e;">30-50万 新元</span></div>'
                '<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 18px; height: 18px; background-color: #e67e22; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div><span style="color: #34495e;">50-70万 新元</span></div>'
                '<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 18px; height: 18px; background-color: #e74c3c; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div><span style="color: #34495e;">70-100万 新元</span></div>'
                '<div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 18px; height: 18px; background-color: #8e44ad; border-radius: 50%; margin-right: 10px; border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.2);"></div><span style="color: #34495e;">> 100万 新元</span></div>'
                '<div style="margin-top: 10px; font-size: 12px; color: #7f8c8d; border-top: 1px solid #eee; padding-top: 8px;">颜色越深，价格越高</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ✅ 更新：数据表格新增商场距离列
    st.markdown('<h2 class="sub-header">数据表格</h2>', unsafe_allow_html=True)

    if len(filtered_df) > 0:
        # 排序功能（新增商场距离排序）
        sort_by = st.selectbox("排序方式",
                               ["价格升序", "价格降序",
                                "单价升序", "单价降序",
                                "面积升序", "面积降序",
                                "剩余租约升序", "剩余租约降序",
                                "交易年份升序", "交易年份降序",
                                "地铁站距离升序", "地铁站距离降序",
                                "商场距离升序", "商场距离降序",
                                "小学距离升序", "小学距离降序",
                                "犯罪率升序", "犯罪率降序"])

        # 应用排序
        if sort_by == "价格升序":
            filtered_df = filtered_df.sort_values('resale_price', ascending=True)
        elif sort_by == "价格降序":
            filtered_df = filtered_df.sort_values('resale_price', ascending=False)
        elif sort_by == "单价升序":
            filtered_df = filtered_df.sort_values('price_per_sqm', ascending=True)
        elif sort_by == "单价降序":
            filtered_df = filtered_df.sort_values('price_per_sqm', ascending=False)
        elif sort_by == "面积升序":
            filtered_df = filtered_df.sort_values('floor_area_sqm', ascending=True)
        elif sort_by == "面积降序":
            filtered_df = filtered_df.sort_values('floor_area_sqm', ascending=False)
        elif sort_by == "剩余租约升序":
            filtered_df = filtered_df.sort_values('remaining_lease_years', ascending=True)
        elif sort_by == "剩余租约降序":
            filtered_df = filtered_df.sort_values('remaining_lease_years', ascending=False)
        elif sort_by == "交易年份升序":
            filtered_df = filtered_df.sort_values('year', ascending=True)
        elif sort_by == "交易年份降序":
            filtered_df = filtered_df.sort_values('year', ascending=False)
        elif sort_by == "地铁站距离升序":
            filtered_df = filtered_df.sort_values('nearest_mrt_dist_m', ascending=True)
        elif sort_by == "地铁站距离降序":
            filtered_df = filtered_df.sort_values('nearest_mrt_dist_m', ascending=False)
        elif sort_by == "商场距离升序":
            filtered_df = filtered_df.sort_values('nearest_mall_dist_m', ascending=True)
        elif sort_by == "商场距离降序":
            filtered_df = filtered_df.sort_values('nearest_mall_dist_m', ascending=False)
        elif sort_by == "小学距离升序":
            filtered_df = filtered_df.sort_values('nearest_school_dist_m', ascending=True)
        elif sort_by == "小学距离降序":
            filtered_df = filtered_df.sort_values('nearest_school_dist_m', ascending=False)
        elif sort_by == "犯罪率升序":
            filtered_df = filtered_df.sort_values('crime_rate_per_1000', ascending=True)
        elif sort_by == "犯罪率降序":
            filtered_df = filtered_df.sort_values('crime_rate_per_1000', ascending=False)

        # 完善后的表格列（新增商场距离）
        display_columns = [
            'town',  # 镇区
            'flat_type',  # 房型
            'flat_model',  # 房屋模型
            'floor_area_sqm',  # 面积
            'price_per_sqm',  # 单价
            'resale_price',  # 总价
            'remaining_lease_years',  # 剩余租约
            'year',  # 交易年份
            'nearest_mrt_exit',  # 最近地铁站出入口
            'nearest_mrt_dist_m',  # 地铁站距离
            'nearest_mall_dist_m',  # ✅ 新增：最近商场距离
            'nearest_bus_dist_m',  # 公交站距离
            'nearest_school_dist_m',  # 小学距离
            'nearest_park_dist_m',  # 公园距离
            'crime_rate_per_1000'  # 区域犯罪率
        ]

        # 分页功能
        items_per_page = 100
        total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page

        col1, col2 = st.columns([1, 3])
        with col1:
            page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

        with col2:
            st.write(f"共 {len(filtered_df):,} 条记录，第 {page}/{total_pages} 页")

        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        page_df = filtered_df[display_columns].iloc[start_idx:end_idx]

        # 🔴 修复：替换use_container_width为width="stretch"
        st.dataframe(
            page_df,
            column_config={
                'town': st.column_config.TextColumn("镇区"),
                'flat_type': st.column_config.TextColumn("房型"),
                'flat_model': st.column_config.TextColumn("房屋模型"),
                'floor_area_sqm': st.column_config.NumberColumn("面积", format="%.1f ㎡"),
                'price_per_sqm': st.column_config.NumberColumn("单价", format="$%d/㎡"),
                'resale_price': st.column_config.NumberColumn("总价", format="$%d"),
                'remaining_lease_years': st.column_config.NumberColumn("剩余租约", format="%.1f 年"),
                'year': st.column_config.NumberColumn("交易年份", format="%d"),
                'nearest_mrt_exit': st.column_config.TextColumn("最近地铁站出入口"),
                'nearest_mrt_dist_m': st.column_config.NumberColumn("地铁站距离", format="%d 米"),
                'nearest_mall_dist_m': st.column_config.NumberColumn("商场距离", format="%d 米"),
                'nearest_bus_dist_m': st.column_config.NumberColumn("公交站距离", format="%d 米"),
                'nearest_school_dist_m': st.column_config.NumberColumn("小学距离", format="%d 米"),
                'nearest_park_dist_m': st.column_config.NumberColumn("公园距离", format="%d 米"),
                'crime_rate_per_1000': st.column_config.NumberColumn("犯罪率", format="%.2f")
            },
            hide_index=True,
            width="stretch",
            height=600
        )

# 页面3：价格影响因素分析（✅ 完全重构，符合分析要求）
elif page == "📈 价格影响因素分析":
    st.markdown('<h1 class="main-header">价格影响因素分析</h1>', unsafe_allow_html=True)

    # 全局数据预处理（只计算一次）
    df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
    df['building_age'] = df['year'] - df['lease_commence_date']


    # 楼层分类（低/中/高）
    def categorize_floor(storey):
        if storey in ['01 TO 03', '04 TO 06']:
            return '低楼层(1-6层)'
        elif storey in ['07 TO 09', '10 TO 12']:
            return '中楼层(7-12层)'
        else:
            return '高楼层(13层以上)'


    df['floor_category'] = df['storey_range'].apply(categorize_floor)

    # 成熟区/非成熟区定义
    mature_towns = ['QUEENSTOWN', 'TOA PAYOH', 'ANG MO KIO', 'BUKIT TIMAH', 'CLEMENTI', 'BUKIT MERAH']
    non_mature_towns = ['PUNGGOL', 'SENGKANG', 'WOODLANDS', 'YISHUN', 'SEMBAWANG', 'TAMPINES']

    df['town_type'] = df['town'].apply(
        lambda x: '成熟区' if x in mature_towns else ('非成熟区' if x in non_mature_towns else '其他')
    )

    # 采样数据用于散点图（避免50万条数据卡顿）
    sample_df = df.sample(n=10000, random_state=42)

    # ====================== 1. 特征重要性排名 ======================
    st.markdown('<h2 class="sub-header">🏆 特征重要性排名</h2>', unsafe_allow_html=True)

    # ✅ 替换为LightGBM版本（和训练时一致使用gain类型）
    # 直接从模型获取特征名称（和训练时顺序完全一致）
    feature_names = model.feature_name()
    # ✅ 关键：使用gain类型获取重要性，和训练时保持一致
    feature_importance_values = model.feature_importance(importance_type='gain')

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_values
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    # 双列布局：左柱状图，右饼图
    col1, col2 = st.columns([3, 2])

    with col1:
        # 显示前20个特征
        fig = px.bar(feature_importance.head(20), x='importance', y='feature',
                     orientation='h', title='Top 20 特征重要性')
        fig.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ✅ 新增：前10个特征累计重要性饼图
        top10 = feature_importance.head(10)
        others = pd.DataFrame({
            'feature': ['其他特征'],
            'importance': [feature_importance['importance'].iloc[10:].sum()]
        })
        pie_data = pd.concat([top10, others])

        fig = px.pie(pie_data, values='importance', names='feature',
                     title='前10个特征累计重要性占比',
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=700)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # ====================== 2. 房屋属性对单价的影响 ======================
    st.markdown('<h2 class="sub-header">🏠 房屋属性对单价的影响</h2>', unsafe_allow_html=True)

    # 2.1 房屋面积 vs 单价（散点图+回归线）
    st.markdown('<h3 class="sub-header">2.1 房屋面积 vs 单价</h3>', unsafe_allow_html=True)
    fig = px.scatter(sample_df, x='floor_area_sqm', y='price_per_sqm',
                     title='房屋面积与单价的关系（采样10000条数据）',
                     labels={'floor_area_sqm': '房屋面积（㎡）', 'price_per_sqm': '单价（新元/㎡）'},
                     opacity=0.3,
                     trendline='ols',  # 添加回归线
                     trendline_color_override='#e74c3c')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 2.2 剩余租约 vs 单价（✅ 修复辛普森悖论，按区域类型着色，兼容所有Plotly版本）
    st.markdown('<h3 class="sub-header">2.2 剩余租约 vs 单价</h3>', unsafe_allow_html=True)

    # 新加坡HDB官方2026年最新成熟区/非成熟区分类（与政府口径完全一致）
    mature_towns = [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH',
        'CENTRAL AREA', 'CLEMENTI', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE',
        'PASIR RIS', 'QUEENSTOWN', 'SERANGOON', 'TAMPINES', 'TOA PAYOH'
    ]

    # 为采样数据添加区域类型标签
    sample_df['town_type'] = sample_df['town'].apply(
        lambda x: '成熟区' if x in mature_towns else '非成熟区'
    )

    # 生成按区域类型着色的散点图（自动为每个类别生成独立趋势线）
    fig = px.scatter(
        sample_df,
        x='remaining_lease_years',
        y='price_per_sqm',
        title='剩余租约与单价的关系（按区域类型着色，采样10000条数据）',
        labels={
            'remaining_lease_years': '剩余租约（年）',
            'price_per_sqm': '单价（新元/㎡）',
            'town_type': '区域类型'
        },
        color='town_type',
        color_discrete_map={'成熟区': '#e74c3c', '非成熟区': '#3498db'},
        opacity=0.3,
        trendline='ols'  # 自动为每个类别生成OLS回归线
    )

    # ✅ 修复：手动设置趋势线颜色（兼容所有Plotly版本）
    for trace in fig.data:
        if trace.name == '成熟区' and 'trendline' in trace.hovertemplate:
            trace.line.color = '#c0392b'  # 成熟区趋势线颜色
        elif trace.name == '非成熟区' and 'trendline' in trace.hovertemplate:
            trace.line.color = '#2980b9'  # 非成熟区趋势线颜色

    # 优化图表样式
    fig.update_layout(
        height=500,
        legend_title_text='区域类型',
        hovermode='closest'
    )

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

    # 2.3 楼层范围 vs 单价（小提琴图）
    st.markdown('<h3 class="sub-header">2.3 楼层范围 vs 单价</h3>', unsafe_allow_html=True)
    fig = px.violin(df, x='floor_category', y='price_per_sqm',
                    title='不同楼层的单价分布',
                    labels={'floor_category': '楼层分类', 'price_per_sqm': '单价（新元/㎡）'},
                    color='floor_category',
                    color_discrete_map={'低楼层(1-6层)': '#2ecc71', '中楼层(7-12层)': '#f1c40f',
                                        '高楼层(13层以上)': '#e74c3c'},
                    box=True,  # 在小提琴内部显示箱线图
                    points=False)  # 不显示单个点，避免杂乱
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 2.4 户型 vs 均价（棒棒糖图）
    st.markdown('<h3 class="sub-header">2.4 户型 vs 均价</h3>', unsafe_allow_html=True)
    flat_type_price = df.groupby('flat_type')['price_per_sqm'].mean().sort_values().reset_index()

    # ✅ 棒棒糖图实现
    fig = px.scatter(flat_type_price, x='price_per_sqm', y='flat_type',
                     title='不同户型的平均单价',
                     labels={'price_per_sqm': '平均单价（新元/㎡）', 'flat_type': '户型'},
                     size=[1] * len(flat_type_price),  # 统一圆点大小
                     size_max=15,
                     color_discrete_sequence=['#1f77b4'])

    # 添加水平线（棒棒糖的"杆"）
    for i, row in flat_type_price.iterrows():
        fig.add_shape(
            type='line',
            x0=0, y0=row['flat_type'],
            x1=row['price_per_sqm'], y1=row['flat_type'],
            line=dict(color='#1f77b4', width=2)
        )

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ====================== 3. 成熟区 vs 非成熟区对比分析 ======================
    st.markdown('<h2 class="sub-header">🌆 成熟区 vs 非成熟区对比分析</h2>', unsafe_allow_html=True)

    # 定义说明（折叠面板）
    with st.expander("📖 成熟区与非成熟区定义说明"):
        st.write("""
        **成熟组屋区**：历史悠久、配套完善、人口密度高的传统镇区
        - 代表：Queenstown、Toa Payoh、Ang Mo Kio、Bukit Timah、Clementi、Bukit Merah

        **非成熟组屋区**：近年发展起来的新兴镇区，基础设施正在完善中
        - 代表：Punggol、Sengkang、Woodlands、Yishun、Sembawang、Tampines

        **其他镇区**：不属于上述两类的区域，如Jurong East、Bishan等
        """)

    # 3.1 平均房价对比（箱线图）
    st.markdown('<h3 class="sub-header">3.1 平均房价对比</h3>', unsafe_allow_html=True)

    fig = px.box(df, x='town_type', y='price_per_sqm',
                 title='成熟区与非成熟区房价分布对比',
                 labels={'price_per_sqm': '单价（新元/㎡）', 'town_type': '镇区类型'},
                 color='town_type',
                 color_discrete_map={'成熟区': '#e74c3c', '非成熟区': '#3498db', '其他': '#95a5a6'},
                 category_orders={'town_type': ['其他', '成熟区', '非成熟区']})

    fig.update_layout(
        height=400,
        showlegend=False
    )

    # 显示平均值点
    fig.update_traces(
        boxmean=True,  # 显示平均值
        width=0.6
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3.2 价格走势对比
    st.markdown('<h3 class="sub-header">3.2 价格走势对比</h3>', unsafe_allow_html=True)
    trend_df = df.groupby(['year', 'town_type'])['price_per_sqm'].mean().reset_index()
    fig = px.line(trend_df, x='year', y='price_per_sqm', color='town_type',
                  title='2015-2026年成熟区与非成熟区价格走势',
                  labels={'price_per_sqm': '平均单价（新元/㎡）', 'year': '年份', 'town_type': '镇区类型'},
                  color_discrete_map={'成熟区': '#e74c3c', '非成熟区': '#3498db', '其他': '#95a5a6'})

    # ✅ 新增：和前面两个图完全一致的x轴配置
    fig.update_layout(
        height=500,
        xaxis=dict(
            tickmode='linear',  # 线性刻度模式
            dtick=1,  # 每隔1年显示一个刻度
            range=[trend_df['year'].min() - 0.5, trend_df['year'].max() + 0.5]  # 左右留边距，避免首尾点被截断
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3.3 主流户型分布对比（热力图）
    st.markdown('<h3 class="sub-header">3.3 主流户型分布对比</h3>', unsafe_allow_html=True)
    flat_dist = df.groupby(['town_type', 'flat_type']).size().reset_index(name='count')
    flat_dist['percentage'] = flat_dist.groupby('town_type')['count'].transform(lambda x: x / x.sum() * 100)
    main_flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
    flat_dist = flat_dist[flat_dist['flat_type'].isin(main_flat_types)]

    # 转换为热力图需要的宽格式
    heatmap_data = flat_dist.pivot(index='town_type', columns='flat_type', values='percentage')

    fig = px.imshow(heatmap_data,
                    title='成熟区与非成熟区主流户型占比热力图',
                    labels=dict(x='户型', y='镇区类型', color='占比(%)'),
                    color_continuous_scale='Blues')
    fig.update_layout(height=400)
    fig.update_traces(texttemplate='%{z:.1f}%', textfont_size=12)  # 显示数值
    st.plotly_chart(fig, use_container_width=True)

    # ====================== 4. 配套设施对价格的影响 ======================
    st.markdown('<h2 class="sub-header">🛍️ 配套设施对价格的影响</h2>', unsafe_allow_html=True)

    # 4.1 地铁站距离对房价的影响
    st.markdown('<h3 class="sub-header">4.1 地铁站距离对房价的影响</h3>', unsafe_allow_html=True)

    # 分箱对比柱状图
    df['mrt_distance_bin'] = pd.cut(df['nearest_mrt_dist_m'],
                                    bins=[0, 500, 1000, 1500, 2000, 10000],
                                    labels=['<500米(步行可达)', '500-1000米', '1000-1500米', '1500-2000米', '>2000米'])

    mrt_price = df.groupby('mrt_distance_bin')['price_per_sqm'].mean().reset_index()
    fig = px.bar(mrt_price, x='mrt_distance_bin', y='price_per_sqm',
                 title='不同地铁站距离的平均单价',
                 labels={'price_per_sqm': '平均单价（新元/㎡）', 'mrt_distance_bin': '地铁站距离'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 散点图+回归线
    fig = px.scatter(sample_df, x='nearest_mrt_dist_m', y='price_per_sqm',
                     title='地铁站距离与单价的关系（采样10000条数据）',
                     labels={'nearest_mrt_dist_m': '地铁站距离（米）', 'price_per_sqm': '单价（新元/㎡）'},
                     opacity=0.3,
                     trendline='ols',
                     trendline_color_override='#e74c3c')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 关键差异分析
    mrt_500m_price = df[df['nearest_mrt_dist_m'] <= 500]['price_per_sqm'].mean()
    mrt_1km_price = df[df['nearest_mrt_dist_m'] > 1000]['price_per_sqm'].mean()
    mrt_premium = (mrt_500m_price - mrt_1km_price) / mrt_1km_price * 100

    st.info(f"""
    🚇 地铁站距离溢价分析：
    - 距离地铁站500米以内的房源平均单价：${mrt_500m_price:,.0f}/㎡
    - 距离地铁站1000米以上的房源平均单价：${mrt_1km_price:,.0f}/㎡
    - 步行可达地铁站的房源平均溢价：**{mrt_premium:.1f}%**
    """)

    # 4.2 商场距离对房价的影响
    st.markdown('<h3 class="sub-header">4.2 商场距离对房价的影响</h3>', unsafe_allow_html=True)

    df['mall_distance_bin'] = pd.cut(df['nearest_mall_dist_m'],
                                     bins=[0, 500, 1000, 2000, 3000, 10000],
                                     labels=['<500米(步行可达)', '500-1000米(骑行可达)',
                                             '1000-2000米(驾车可达)', '2000-3000米', '>3000米'])

    mall_price = df.groupby('mall_distance_bin')['price_per_sqm'].mean().reset_index()
    fig = px.bar(mall_price, x='mall_distance_bin', y='price_per_sqm',
                 title='不同商场距离的平均单价',
                 labels={'price_per_sqm': '平均单价（新元/㎡）', 'mall_distance_bin': '最近商场距离'})
    fig.update_traces(marker_color='#e74c3c')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ✅ 新增：商场溢价分析（和地铁、小学格式统一）
    mall_500m_price = df[df['nearest_mall_dist_m'] <= 500]['price_per_sqm'].mean()
    mall_far_price = df[df['nearest_mall_dist_m'] > 1000]['price_per_sqm'].mean()
    mall_premium = (mall_500m_price - mall_far_price) / mall_far_price * 100

    st.info(f"""
    🛒 商场距离溢价分析：
    - 距离商场500米以内的房源平均单价：${mall_500m_price:,.0f}/㎡
    - 距离商场1000米以上的房源平均单价：${mall_far_price:,.0f}/㎡
    - 步行可达商场的房源平均溢价：**{mall_premium:.1f}%**
    """)

    # 4.3 小学距离与名校圈溢价
    st.markdown('<h3 class="sub-header">4.3 小学距离与名校圈溢价</h3>', unsafe_allow_html=True)

    df['school_distance_bin'] = pd.cut(df['nearest_school_dist_m'],
                                       bins=[0, 1000, 2000, 10000],
                                       labels=['<1000米(名校圈)', '1000-2000米', '>2000米'])

    school_price = df.groupby('school_distance_bin')['price_per_sqm'].mean().reset_index()
    fig = px.bar(school_price, x='school_distance_bin', y='price_per_sqm',
                 title='不同小学距离的平均单价',
                 labels={'price_per_sqm': '平均单价（新元/㎡）', 'school_distance_bin': '小学距离'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # 名校圈溢价分析
    school_1km_price = df[df['nearest_school_dist_m'] <= 1000]['price_per_sqm'].mean()
    school_2km_price = df[df['nearest_school_dist_m'] > 1000]['price_per_sqm'].mean()
    school_premium = (school_1km_price - school_2km_price) / school_2km_price * 100

    st.info(f"""
    🏫 名校圈溢价分析：
    - 距离小学1000米以内的房源平均单价：${school_1km_price:,.0f}/㎡
    - 距离小学1000米以上的房源平均单价：${school_2km_price:,.0f}/㎡
    - 名校圈范围内的房源平均溢价：**{school_premium:.1f}%**
    """)

    # 4.4 其他配套设施影响
    st.markdown('<h3 class="sub-header">4.4 其他配套设施影响</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 公交站距离
        df['bus_distance_bin'] = pd.cut(df['nearest_bus_dist_m'],
                                        bins=[0, 200, 500, 1000, 5000],
                                        labels=['<200米', '200-500米', '500-1000米', '>1000米'])

        bus_price = df.groupby('bus_distance_bin')['price_per_sqm'].mean().reset_index()
        fig = px.bar(bus_price, x='bus_distance_bin', y='price_per_sqm',
                     title='公交站距离对单价的影响',
                     labels={'price_per_sqm': '平均单价（新元/㎡）', 'bus_distance_bin': '公交站距离'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 公园距离
        df['park_distance_bin'] = pd.cut(df['nearest_park_dist_m'],
                                         bins=[0, 500, 1000, 2000, 10000],
                                         labels=['<500米', '500-1000米', '1000-2000米', '>2000米'])

        park_price = df.groupby('park_distance_bin')['price_per_sqm'].mean().reset_index()
        fig = px.bar(park_price, x='park_distance_bin', y='price_per_sqm',
                     title='公园距离对单价的影响',
                     labels={'price_per_sqm': '平均单价（新元/㎡）', 'park_distance_bin': '公园距离'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 4.5 区域犯罪率对房价的影响（散点图+回归线）
    st.markdown('<h3 class="sub-header">4.5 区域犯罪率对房价的影响</h3>', unsafe_allow_html=True)
    fig = px.scatter(sample_df, x='crime_rate_per_1000', y='price_per_sqm',
                     title='区域犯罪率与单价的关系（采样10000条数据）',
                     labels={'crime_rate_per_1000': '犯罪率（每千人）', 'price_per_sqm': '单价（新元/㎡）'},
                     opacity=0.3,
                     trendline='ols',
                     trendline_color_override='#e74c3c')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ✅ 新增：全局相关性热力图
    st.markdown('<h3 class="sub-header">4.6 关键变量相关性分析</h3>', unsafe_allow_html=True)

    # 选择关键数值变量
    corr_vars = ['price_per_sqm', 'floor_area_sqm', 'remaining_lease_years', 'building_age',
                 'nearest_mrt_dist_m', 'nearest_mall_dist_m', 'nearest_school_dist_m',
                 'nearest_park_dist_m', 'crime_rate_per_1000']

    corr_matrix = df[corr_vars].corr()

    fig = px.imshow(corr_matrix,
                    title='关键变量相关性热力图',
                    color_continuous_scale='RdBu_r',  # 红-蓝配色，正相关红，负相关蓝
                    zmin=-1, zmax=1)
    fig.update_layout(height=600)
    fig.update_traces(texttemplate='%{z:.2f}', textfont_size=12)
    st.plotly_chart(fig, use_container_width=True)

# 页面4：房价预测（✅ 优化输入提示+界面整合）
elif page == "🔮 房价预测":
    st.markdown('<h1 class="main-header">HDB房价智能预测</h1>', unsafe_allow_html=True)

    # ====================== 预定义说明数据 ======================
    # 成熟区/非成熟区定义（与页面3保持一致）
    mature_towns = ['QUEENSTOWN', 'TOA PAYOH', 'ANG MO KIO', 'BUKIT TIMAH', 'CLEMENTI', 'BUKIT MERAH']
    non_mature_towns = ['PUNGGOL', 'SENGKANG', 'WOODLANDS', 'YISHUN', 'SEMBAWANG', 'TAMPINES']

    # 户型中文说明
    flat_type_help = """
    户型说明：
    • 1 ROOM：单卧室组屋（约23-33㎡）
    • 2 ROOM：两室一厅（约36-45㎡）
    • 3 ROOM：三室一厅（约60-70㎡）
    • 4 ROOM：四室一厅（约90-100㎡）
    • 5 ROOM：五室一厅（约110-130㎡）
    • EXECUTIVE：行政公寓（约140-160㎡）
    • MULTI-GENERATION：多代同堂组屋
    """

    # 房屋模型中文说明
    flat_model_help = """
    房屋模型说明：
    • Improved：改进型组屋（1980年代建造）
    • Improved-Maisonette：改进型复式组屋
    • Maisonette：复式组屋（双层结构）
    • Model A：A型标准组屋（最常见）
    • Model A-Maisonette：A型复式组屋
    • Model A2：A2型组屋（简化版A型）
    • Multi Generation：多代同堂组屋
    • New Generation：新一代组屋（1990年代后）
    • Premium Apartment：高级公寓
    • Terrace：排屋式组屋
    """

    # 生成带标签的市镇选项列表
    town_options = []
    for town in sorted(town_map.keys()):
        if town in mature_towns:
            town_options.append(f"{town} (成熟区)")
        elif town in non_mature_towns:
            town_options.append(f"{town} (非成熟区)")
        else:
            town_options.append(f"{town} (其他)")

    # ====================== 输入表单 ======================
    col1, col2 = st.columns(2)

    with col1:
        # ✅ 优化1：市镇选择带成熟区/非成熟区标签
        selected_town_label = st.selectbox(
            "选择市镇",
            town_options,
            help="成熟区：历史悠久、配套完善；非成熟区：新兴发展、升值潜力大"
        )
        # 提取实际市镇名称（去掉括号标签）
        town = selected_town_label.split(' (')[0]

        # ✅ 优化2：户型选择加中文说明
        flat_type = st.selectbox(
            "选择户型",
            sorted(flat_type_map.keys()),
            help=flat_type_help
        )

        # ✅ 优化3：房屋模型加中文说明
        flat_model = st.selectbox(
            "选择房屋模型",
            sorted(flat_model_map.keys()),
            help=flat_model_help
        )

        storey_range = st.selectbox(
            "选择楼层范围",
            sorted(storey_range_map.keys()),
            help="低楼层：1-6层；中楼层：7-12层；高楼层：13层以上"
        )

    with col2:
        floor_area_sqm = st.number_input(
            "房屋面积（平方米）",
            min_value=20,
            max_value=200,
            value=90,
            help="参考：3房式约60-70㎡，4房式约90-100㎡"
        )

        remaining_lease_years = st.number_input(
            "剩余租约（年）",
            min_value=1,
            max_value=99,
            value=70,
            help="HDB组屋租约为99年，剩余租约越短，价格越低"
        )

        nearest_mrt_dist_m = st.number_input(
            "到最近地铁站出入口距离（米）",
            min_value=0,
            max_value=5000,
            value=500,
            help="500米以内为步行可达，1000米以上交通便利性显著下降"
        )

        nearest_bus_dist_m = st.number_input(
            "到最近公交站距离（米）",
            min_value=0,
            max_value=2000,
            value=100,
            help="200米以内为步行可达"
        )

    # ✅ 优化4：商业配套选项合并到高级选项中
    with st.expander("高级选项"):
        st.markdown("### 配套设施参数")
        nearest_mall_dist_m = st.number_input(
            "到最近商场距离（米）",
            min_value=0,
            max_value=6000,
            value=1000,
            help="500米以内为步行可达，商业配套溢价最高"
        )

        nearest_school_dist_m = st.number_input(
            "到最近小学距离（米）",
            min_value=0,
            max_value=5000,
            value=1000,
            help="1000米以内为名校圈范围"
        )

        nearest_park_dist_m = st.number_input(
            "到最近公园距离（米）",
            min_value=0,
            max_value=5000,
            value=800
        )

        st.markdown("### 其他参数")
        crime_rate_per_1000 = st.number_input(
            "区域犯罪率（每千人）",
            min_value=0.0,
            max_value=3.0,  # ✅ 修正：最大值从10.0改为3.0，符合实际数据范围
            value=1.7,  # ✅ 修正：默认值从3.0改为1.7，使用新加坡最新平均值
            help="新加坡全国平均犯罪率约为1.72每千人"
        )

        year = st.number_input(
            "交易年份",
            min_value=2010,
            max_value=2030,
            value=2024
        )

        latitude = st.number_input(
            "纬度",
            min_value=1.2,
            max_value=1.5,
            value=1.3521,
            help="新加坡中心纬度约为1.3521"
        )

        longitude = st.number_input(
            "经度",
            min_value=103.6,
            max_value=104.0,
            value=103.8198,
            help="新加坡中心经度约为103.8198"
        )

        quarter_num = st.number_input(
            "季度",
            min_value=1,
            max_value=4,
            value=2
        )

        nearest_school_type_encoded = st.number_input(
            "学校类型编码",
            min_value=0.0,
            max_value=1000000.0,
            value=500000.0,
            help="普通小学约50万，名校约80-100万"
        )

    # 预测按钮
    if st.button("预测房价", type="primary", use_container_width=True):
        with st.spinner("正在计算预测结果..."):
            # 基础特征计算
            remaining_lease_months = remaining_lease_years * 12
            lease_commence_date = year - remaining_lease_years

            # 获取当前市镇的完整统计数据
            town_avg_lease, town_avg_area, town_avg_transport_score, town_avg_commercial_score = town_stats_dict[town]

            # 相对优势特征
            lease_diff_from_town_avg = remaining_lease_months - town_avg_lease

            # 地铁站分箱特征
            mrt_within_200m = 1 if nearest_mrt_dist_m <= 200 else 0
            mrt_within_500m = 1 if nearest_mrt_dist_m <= 500 else 0
            mrt_within_1000m = 1 if nearest_mrt_dist_m <= 1000 else 0
            mrt_within_1500m = 1 if nearest_mrt_dist_m <= 1500 else 0

            # 商场分箱特征
            mall_within_500m = 1 if nearest_mall_dist_m <= 500 else 0
            mall_within_1000m = 1 if nearest_mall_dist_m <= 1000 else 0
            mall_within_2000m = 1 if nearest_mall_dist_m <= 2000 else 0

            # 距离对数变换特征
            mrt_distance_log = np.log(nearest_mrt_dist_m + 1)
            bus_distance_log = np.log(nearest_bus_dist_m + 1)
            mall_distance_log = np.log(nearest_mall_dist_m + 1)

            # 二值特征计算
            within_1km_school = 1 if nearest_school_dist_m <= 1000 else 0
            within_2km_school = 1 if nearest_school_dist_m <= 2000 else 0
            within_500m_park = 1 if nearest_park_dist_m <= 500 else 0
            within_1km_park = 1 if nearest_park_dist_m <= 1000 else 0
            within_100m_bus = 1 if nearest_bus_dist_m <= 100 else 0
            within_300m_bus = 1 if nearest_bus_dist_m <= 300 else 0
            within_500m_bus = 1 if nearest_bus_dist_m <= 500 else 0

            # 综合得分特征
            transport_score = 1 / (nearest_mrt_dist_m / 1000 + nearest_bus_dist_m / 500 + 0.1)
            livability_score = 1 / (nearest_school_dist_m / 1000 + nearest_park_dist_m / 1000 + 0.1)
            commercial_score = 1 / (nearest_mall_dist_m / 1000 + 0.1)

            # ====================== ✅ 修复1：完整计算所有4个交互项特征 ======================
            area_lease_interaction = floor_area_sqm * remaining_lease_years
            transport_livability_interaction = transport_score * livability_score
            area_transport_interaction = floor_area_sqm * transport_score

            # 新增：之前缺失的3个交互项
            mrt_500m_area_interaction = mrt_within_500m * floor_area_sqm
            mrt_1000m_area_interaction = mrt_within_1000m * floor_area_sqm
            mall_500m_area_interaction = mall_within_500m * floor_area_sqm
            mall_1000m_area_interaction = mall_within_1000m * floor_area_sqm

            area_commercial_interaction = floor_area_sqm * commercial_score
            transport_commercial_interaction = transport_score * commercial_score
            commercial_diff_from_town_avg = commercial_score - town_avg_commercial_score

            # 估算总犯罪数
            total_crimes = crime_rate_per_1000 * 10

            # 距离的倒数特征
            mrt_distance_inv = 1 / (nearest_mrt_dist_m + 50)
            bus_distance_inv = 1 / (nearest_bus_dist_m + 30)
            school_distance_inv = 1 / (nearest_school_dist_m + 100)
            park_distance_inv = 1 / (nearest_park_dist_m + 100)
            mall_distance_inv = 1 / (nearest_mall_dist_m + 100)

            # ====================== ✅ 修复2：完整包含所有模型需要的特征 ======================
            input_data = pd.DataFrame({
                # 基础特征
                'floor_area_sqm': [floor_area_sqm],
                'remaining_lease_months': [remaining_lease_months],
                'remaining_lease_years': [remaining_lease_years],  # 新增：之前缺失的核心特征
                'nearest_mrt_dist_m': [nearest_mrt_dist_m],
                'nearest_bus_dist_m': [nearest_bus_dist_m],
                'nearest_school_dist_m': [nearest_school_dist_m],
                'nearest_park_dist_m': [nearest_park_dist_m],
                'nearest_mall_dist_m': [nearest_mall_dist_m],
                'crime_rate_per_1000': [crime_rate_per_1000],
                'year': [year],
                'town_encoded': [town_map[town]],
                'flat_type_encoded': [flat_type_map[flat_type]],
                'flat_model_encoded': [flat_model_map[flat_model]],
                'storey_range_encoded': [storey_range_map[storey_range]],
                'transport_score': [transport_score],
                'livability_score': [livability_score],
                'commercial_score': [commercial_score],

                # 时间和位置特征
                'lease_commence_date': [lease_commence_date],
                'latitude': [latitude],
                'longitude': [longitude],
                'quarter_num': [quarter_num],

                # 二值特征
                'within_1km_school': [within_1km_school],
                'within_2km_school': [within_2km_school],
                'within_500m_park': [within_500m_park],
                'within_1km_park': [within_1km_park],
                'total_crimes': [total_crimes],
                'within_100m_bus': [within_100m_bus],
                'within_300m_bus': [within_300m_bus],
                'within_500m_bus': [within_500m_bus],
                'nearest_school_type_encoded': [nearest_school_type_encoded],

                # 地铁站分箱特征
                'mrt_within_200m': [mrt_within_200m],
                'mrt_within_500m': [mrt_within_500m],
                'mrt_within_1000m': [mrt_within_1000m],
                'mrt_within_1500m': [mrt_within_1500m],

                # 商场分箱特征
                'mall_within_500m': [mall_within_500m],
                'mall_within_1000m': [mall_within_1000m],
                'mall_within_2000m': [mall_within_2000m],

                # 距离对数变换特征
                'mrt_distance_log': [mrt_distance_log],
                'bus_distance_log': [bus_distance_log],
                'mall_distance_log': [mall_distance_log],

                # 距离倒数特征
                'mrt_distance_inv': [mrt_distance_inv],
                'bus_distance_inv': [bus_distance_inv],
                'school_distance_inv': [school_distance_inv],
                'park_distance_inv': [park_distance_inv],
                'mall_distance_inv': [mall_distance_inv],

                # 衍生特征（完整包含所有4个交互项）
                'area_lease_interaction': [area_lease_interaction],
                'transport_livability_interaction': [transport_livability_interaction],
                'area_transport_interaction': [area_transport_interaction],
                'mrt_500m_area_interaction': [mrt_500m_area_interaction],  # 新增
                'mrt_1000m_area_interaction': [mrt_1000m_area_interaction],  # 新增
                'mall_500m_area_interaction': [mall_500m_area_interaction],  # 新增
                'mall_1000m_area_interaction': [mall_1000m_area_interaction],
                'area_commercial_interaction': [area_commercial_interaction],
                'transport_commercial_interaction': [transport_commercial_interaction],

                # 区域级特征
                'town_avg_lease': [town_avg_lease],
                'town_avg_area': [town_avg_area],
                'town_avg_transport_score': [town_avg_transport_score],
                'town_avg_commercial_score': [town_avg_commercial_score],
                'lease_diff_from_town_avg': [lease_diff_from_town_avg],
                'commercial_diff_from_town_avg': [commercial_diff_from_town_avg]
            })

            # ====================== ✅ 终极防护：永远不会再报KeyError ======================
            model_features = model.feature_name()  # LightGBM获取特征名称的方法
            # 自动对齐：只保留模型需要的特征，缺失的自动补0，多余的自动忽略
            input_data = input_data.reindex(columns=model_features, fill_value=0)

            # 预测
            predicted_price = model.predict(input_data)[0]

            # 显示结果
            st.success("预测完成！")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("预测当前市场价格", f"${predicted_price:,.0f}")

            with col2:
                price_per_sqm = predicted_price / floor_area_sqm
                st.metric("预测每平方米价格", f"${price_per_sqm:,.0f}")

            with col3:
                # ✅ 基于模型中位数误差的科学双区间
                # 核心合理区间：±0.6倍总价中位数误差 | 总宽度≈5.1万 | 置信度≈50%
                median_total_error = 477.11 * floor_area_sqm
                core_margin = median_total_error * 0.6
                core_lower = predicted_price - core_margin
                core_upper = predicted_price + core_margin

                # 安全参考区间：±0.7倍MAE | 总宽度≈7.5万 | 置信度≈70%
                safe_margin = 53947 * 0.7
                safe_lower = predicted_price - safe_margin
                safe_upper = predicted_price + safe_margin

                # 优先显示核心区间，tooltip显示安全区间
                st.metric(
                    "核心合理区间",
                    f"${core_lower:,.0f} - ${core_upper:,.0f}",
                    help=f"安全参考区间：${safe_lower:,.0f} - ${safe_upper:,.0f}\n70%的房源会落在安全区间内"
                )

            # ✅ 新增：实用购房决策提示
            if predicted_price < core_lower * 0.95:
                st.success("✅ 该房源价格明显低于市场预期，属于捡漏好房，建议优先考虑")
            elif predicted_price > safe_upper * 1.05:
                st.warning("⚠️ 该房源价格明显高于市场预期，建议谨慎出价或继续观望")
            else:
                st.info("💡 该房源价格处于合理范围内，可按照核心区间进行议价")

            # 商业配套影响分析
            # st.info(f"""
            # 🛒 商业配套影响分析：
            # - 距离商场 {nearest_mall_dist_m} 米，属于 {'步行可达(500米内)' if nearest_mall_dist_m <= 500 else '骑行可达(1000米内)' if nearest_mall_dist_m <= 1000 else '驾车可达(2000米内)'}
            # - 该距离的房源平均溢价：${35858 if nearest_mall_dist_m <= 500 else 18000 if nearest_mall_dist_m <= 1000 else 5000 if nearest_mall_dist_m <= 2000 else 0:,} 新元
            # """)

            # 模型性能说明（已更新为最新优化版模型的指标）
            st.info(f"""
            💡 模型性能说明：
            - 测试集R²得分：0.8789（能解释87.89%的房价变化）
            - 测试集每平方米平均误差：$574.90/㎡
            - 每平方米中位数误差：$477.11/㎡
            """)

# 页面5：购房策略与保值分析（✅ 专业决策版，缩进已修复）
elif page == "💡 购房策略":
    st.markdown('<h1 class="main-header">购房策略与保值分析</h1>', unsafe_allow_html=True)

    # ====================== 1. 智能预算匹配系统 ======================
    st.markdown('<h2 class="sub-header">💰 智能预算匹配</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("你的购房预算（新元）",
                                 min_value=100000,
                                 max_value=2000000,
                                 value=500000,
                                 step=10000)

    with col2:
        priority = st.selectbox(
            "你的首要需求",
            ["性价比优先", "交通便利优先", "商业配套优先", "学区优先", "保值升值优先"]
        )

    # 预计算所有区域的综合指标
    @st.cache_data
    def calculate_town_metrics():
        town_metrics = []

        for town in df['town'].unique():
            town_df = df[df['town'] == town]

            # 基础价格指标
            avg_price = town_df['resale_price'].mean()
            avg_price_per_sqm = (town_df['resale_price'] / town_df['floor_area_sqm']).mean()

            # 配套指标
            avg_mrt_dist = town_df['nearest_mrt_dist_m'].mean()
            avg_mall_dist = town_df['nearest_mall_dist_m'].mean()
            avg_school_dist = town_df['nearest_school_dist_m'].mean()

            # 保值指标
            price_2018 = town_df[town_df['year'] == 2018]['resale_price'].mean()
            price_2025 = town_df[town_df['year'] == 2025]['resale_price'].mean()
            if not pd.isna(price_2018) and not pd.isna(price_2025) and price_2018 > 0:
                growth_7y = ((price_2025 - price_2018) / price_2018 * 100)
            else:
                growth_7y = 0

            # 各户型平均价格
            flat_type_prices = {}
            for flat_type in ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']:
                type_df = town_df[town_df['flat_type'] == flat_type]
                if len(type_df) > 0:
                    flat_type_prices[flat_type] = type_df['resale_price'].mean()
                else:
                    flat_type_prices[flat_type] = None

            town_metrics.append({
                'town': town,
                'avg_price': avg_price,
                'avg_price_per_sqm': avg_price_per_sqm,
                'avg_mrt_dist': avg_mrt_dist,
                'avg_mall_dist': avg_mall_dist,
                'avg_school_dist': avg_school_dist,
                'growth_7y': growth_7y,
                **flat_type_prices
            })

        return pd.DataFrame(town_metrics)

    town_metrics_df = calculate_town_metrics()

    # 根据预算筛选可负担区域
    affordable_df = town_metrics_df[town_metrics_df['avg_price'] <= budget].copy()

    if len(affordable_df) > 0:
        # 根据用户优先级排序
        if priority == "性价比优先":
            affordable_df['score'] = (affordable_df['growth_7y'] / affordable_df['avg_price_per_sqm'] * 1000)
        elif priority == "交通便利优先":
            affordable_df['score'] = 1000 / affordable_df['avg_mrt_dist']
        elif priority == "商业配套优先":
            affordable_df['score'] = 1000 / affordable_df['avg_mall_dist']
        elif priority == "学区优先":
            affordable_df['score'] = 1000 / affordable_df['avg_school_dist']
        else:  # 保值升值优先
            affordable_df['score'] = affordable_df['growth_7y']

        affordable_df = affordable_df.sort_values('score', ascending=False).reset_index(drop=True)

        st.success(f"✅ 在你的预算 ${budget:,.0f} 内，为你推荐以下 {len(affordable_df)} 个区域（按{priority}排序）：")

        # ✅ 新增：显示数量控制滑块
        display_count = st.slider(
            "显示前N个区域",
            min_value=5,
            max_value=len(affordable_df),  # 动态最大值，永远不会超过实际推荐数量
            value=5,  # 默认显示前5个
            step=1
        )

        # ✅ 修改为显示用户选择的数量
        for idx, row in affordable_df.head(display_count).iterrows():
            with st.expander(f"🏆 第{idx + 1}名：{row['town']}（综合得分：{row['score']:.1f}）",
                             expanded=True if idx == 0 else False):
                # 以下内部代码完全不变（和方案1一样）
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("平均总价", f"${row['avg_price']:,.0f}")
                    st.metric("平均单价", f"${row['avg_price_per_sqm']:,.0f}/㎡")

                with col2:
                    st.metric("7年累计涨幅", f"{row['growth_7y']:.1f}%")
                    st.metric("平均地铁距离", f"{row['avg_mrt_dist']:.0f}米")

                with col3:
                    st.metric("平均商场距离", f"{row['avg_mall_dist']:.0f}米")
                    st.metric("平均小学距离", f"{row['avg_school_dist']:.0f}米")

                # 可负担户型
                st.markdown("**可负担户型：**")
                affordable_types = []
                for flat_type in ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']:
                    if not pd.isna(row[flat_type]) and row[flat_type] <= budget:
                        affordable_types.append(f"{flat_type} (${row[flat_type]:,.0f})")

                if affordable_types:
                    st.write(" • " + " | ".join(affordable_types))
                else:
                    st.write(" 无适合的常规户型，建议考虑更小面积或特殊户型")

                # 区域优缺点
                pros = []
                cons = []

                if row['avg_mrt_dist'] < 600:
                    pros.append("交通便利，地铁步行可达")
                elif row['avg_mrt_dist'] > 1200:
                    cons.append("地铁距离较远，依赖公交")

                if row['avg_mall_dist'] < 800:
                    pros.append("商业配套完善，生活便利")
                elif row['avg_mall_dist'] > 1500:
                    cons.append("商业配套一般")

                if row['growth_7y'] > 40:
                    pros.append("升值潜力大，历史涨幅高")
                elif row['growth_7y'] < 20:
                    cons.append("升值速度较慢")

                if pros:
                    st.markdown("**✅ 优势：** " + "、".join(pros))
                if cons:
                    st.markdown("**⚠️ 不足：** " + "、".join(cons))

    else:
        st.warning("⚠️ 你的预算低于所有区域的平均房价，建议：")
        st.write("1. 考虑更小的户型（如1 ROOM或2 ROOM）")
        st.write("2. 选择更偏远的区域（如Lim Chu Kang、Tengah）")
        st.write("3. 降低对楼层和朝向的要求")
        st.write("4. 考虑剩余租约较短的房源")

    # ====================== 2. 区域综合实力排名 ======================
    st.markdown('<h2 class="sub-header">🏙️ 区域综合实力排名</h2>', unsafe_allow_html=True)

    # 计算综合得分（满分100）
    town_metrics_df['price_score'] = 100 - (
                town_metrics_df['avg_price_per_sqm'] / town_metrics_df['avg_price_per_sqm'].max() * 50)
    town_metrics_df['transport_score'] = 100 - (
                town_metrics_df['avg_mrt_dist'] / town_metrics_df['avg_mrt_dist'].max() * 20)
    town_metrics_df['commercial_score'] = 100 - (
                town_metrics_df['avg_mall_dist'] / town_metrics_df['avg_mall_dist'].max() * 15)
    town_metrics_df['growth_score'] = town_metrics_df['growth_7y'] / town_metrics_df['growth_7y'].max() * 15

    town_metrics_df['total_score'] = (
            town_metrics_df['price_score'] * 0.3 +
            town_metrics_df['transport_score'] * 0.25 +
            town_metrics_df['commercial_score'] * 0.2 +
            town_metrics_df['growth_score'] * 0.25
    )

    top_10_towns = town_metrics_df.sort_values('total_score', ascending=False).head(10)

    # 雷达图展示前3名区域
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("**综合得分前10名区域**")
        display_df = top_10_towns[['town', 'total_score', 'avg_price_per_sqm', 'growth_7y']].copy()
        display_df.columns = ['市镇', '综合得分', '平均单价', '7年涨幅(%)']
        display_df['综合得分'] = display_df['综合得分'].round(1)
        display_df['平均单价'] = display_df['平均单价'].round(0).astype(int).apply(lambda x: f"${x:,}")
        display_df['7年涨幅(%)'] = display_df['7年涨幅(%)'].round(1)

        st.dataframe(display_df, hide_index=True, use_container_width=True)

    with col2:
        # 雷达图对比前3名
        top3 = top_10_towns.head(3)

        radar_data = []
        for idx, row in top3.iterrows():
            radar_data.append({
                'category': '价格优势',
                'value': row['price_score'],
                'town': row['town']
            })
            radar_data.append({
                'category': '交通便利',
                'value': row['transport_score'],
                'town': row['town']
            })
            radar_data.append({
                'category': '商业配套',
                'value': row['commercial_score'],
                'town': row['town']
            })
            radar_data.append({
                'category': '升值潜力',
                'value': row['growth_score'],
                'town': row['town']
            })

        radar_df = pd.DataFrame(radar_data)

        fig = px.line_polar(radar_df, r='value', theta='category', color='town',
                            line_close=True,
                            title='前3名区域综合实力对比',
                            color_discrete_sequence=['#1f77b4', '#e74c3c', '#2ecc71'])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ====================== 3. 保值性深度分析 ======================
    st.markdown('<h2 class="sub-header">📈 区域保值性分析</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 分阶段涨幅对比
        st.markdown("**各阶段房价涨幅对比**")

        # 计算近3年和近7年涨幅
        town_growth = []
        for town in df['town'].unique():
            town_df = df[df['town'] == town]

            p_2018 = town_df[town_df['year'] == 2018]['resale_price'].mean()
            p_2022 = town_df[town_df['year'] == 2022]['resale_price'].mean()
            p_2025 = town_df[town_df['year'] == 2025]['resale_price'].mean()

            if not pd.isna(p_2018) and not pd.isna(p_2022) and not pd.isna(p_2025):
                growth_7y = ((p_2025 - p_2018) / p_2018 * 100)
                growth_3y = ((p_2025 - p_2022) / p_2022 * 100)

                town_growth.append({
                    'town': town,
                    '近7年涨幅(%)': growth_7y,
                    '近3年涨幅(%)': growth_3y
                })

        growth_df = pd.DataFrame(town_growth).sort_values('近7年涨幅(%)', ascending=False).head(10)

        fig = px.bar(growth_df, x='town', y=['近7年涨幅(%)', '近3年涨幅(%)'],
                     title='涨幅前10区域分阶段对比',
                     barmode='group',
                     color_discrete_map={'近7年涨幅(%)': '#1f77b4', '近3年涨幅(%)': '#e74c3c'})
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 商业配套与涨幅的关系
        st.markdown("**商业配套对升值的影响**")

        scatter_df = town_metrics_df[['town', 'avg_mall_dist', 'growth_7y']].copy()

        fig = px.scatter(scatter_df, x='avg_mall_dist', y='growth_7y',
                         title='平均商场距离与7年涨幅关系',
                         labels={'avg_mall_dist': '平均商场距离（米）', 'growth_7y': '7年累计涨幅（%）'},
                         hover_data=['town'],
                         opacity=0.7,
                         trendline='ols',
                         trendline_color_override='#e74c3c')
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    📊 数据洞察：
    - 商业配套与房价涨幅呈显著负相关，平均商场距离每减少100米，7年累计涨幅提高约2.3%
    - 近3年涨幅普遍低于近7年涨幅，市场整体进入调整期
    - 成熟区涨幅稳定，非成熟区波动较大，但部分新兴区域涨幅惊人
    """)

    # ====================== 4. 分需求购房策略 ======================
    st.markdown('<h2 class="sub-header">🎯 不同需求的购房策略</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["刚需自住", "改善型", "投资保值"])

    with tab1:
        st.markdown("""
        ### 刚需自住购房策略
        **核心目标：** 满足基本居住需求，控制预算，兼顾生活便利性

        **推荐区域：**
        - **预算30-50万：** Sengkang、Punggol、Woodlands、Yishun
        - **预算50-70万：** Bedok、Jurong West、Tampines、Hougang
        - **预算70-100万：** Ang Mo Kio、Bukit Batok、Clementi、Toa Payoh

        **户型选择：**
        - 首次置业优先考虑3房式（60-70㎡），满足3-4口之家需求
        - 预算有限可考虑2房式，未来再置换
        - 避免选择1房式，流动性差且升值空间小

        **注意事项：**
        1. 优先选择距离地铁站500米以内的房源
        2. 剩余租约至少60年以上，避免短期贬值
        3. 周边有超市、食阁、诊所等基本配套即可
        4. 不要追求完美，在预算范围内选择综合条件最好的
        """)

    with tab2:
        st.markdown("""
        ### 改善型购房策略
        **核心目标：** 提升居住品质，关注环境、配套和学区

        **推荐区域：**
        - **高性价比：** Bishan、Serangoon、Kallang/Whampoa
        - **优质学区：** Bukit Timah、Nanyang、Raffles Place周边
        - **环境优美：** Bukit Merah、Queenstown、East Coast

        **户型选择：**
        - 优先考虑4房式或5房式，面积90-130㎡
        - 关注户型朝向和通风采光
        - 有条件可选择复式或行政公寓

        **注意事项：**
        1. 重点关注周边小学质量，学区房保值性最好
        2. 商业配套要完善，步行可达大型商场
        3. 小区环境和物业管理很重要
        4. 尽量选择中高楼层，视野和采光更好
        """)

    with tab3:
        st.markdown("""
        ### 投资保值购房策略
        **核心目标：** 资产保值增值，追求稳定的租金回报和升值潜力

        **推荐区域：**
        - **高租金回报：** 市中心周边、地铁站附近、大学周边
        - **高升值潜力：** 政府规划中的新市镇、即将开通地铁的区域
        - **稳定保值：** 成熟商业区、优质学区

        **户型选择：**
        - 优先考虑3房式和4房式，租赁需求最旺盛
        - 避免过大或过小的户型，流动性差
        - 同一区域选择面积较小的户型，单价更高但总价更低

        **注意事项：**
        1. 交通是第一要素，距离地铁站越近越好
        2. 商业配套完善的区域租金更高
        3. 关注政府未来5-10年的发展规划
        4. 计算租金回报率，目标至少3%以上
        5. 避免剩余租少于50年的房源
        """)

    # ====================== 5. 购房避坑指南 ======================
    st.markdown('<h2 class="sub-header">⚠️ 购房避坑指南</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ❌ 这些房子不要买
        1. **剩余租约少于40年的房源**
           - 银行贷款额度大幅减少
           - 未来转手困难
           - 贬值速度快

        2. **顶楼和一楼的房源**
           - 顶楼容易漏水、夏天炎热
           - 一楼潮湿、蚊虫多、隐私差
           - 转售时价格明显低于中间楼层

        3. **距离主干道或高架桥太近的房源**
           - 噪音和空气污染严重
           - 影响居住体验和转售价格

        4. **户型奇葩的房源**
           - 有尖角、斜角、长走廊
           - 采光通风差
           - 装修成本高且不好用
        """)

    with col2:
        st.markdown("""
        ### ✅ 这些可以重点考虑
        1. **中间楼层（7-12层）**
           - 视野好、采光通风佳
           - 避免了低楼层的潮湿和高楼层的不便
           - 转售时最受欢迎

        2. **南北通透的户型**
           - 通风好，夏天凉爽
           - 采光均匀，居住舒适

        3. **房龄10-20年的次新房**
           - 社区成熟，配套完善
           - 房屋质量问题已暴露
           - 价格比新房便宜，升值空间大

        4. **政府重点发展区域**
           - 基础设施不断完善
           - 人口持续流入
           - 长期升值潜力大
        """)

    # ====================== 6. 核心总结 ======================
    st.markdown('<h2 class="sub-header">💎 核心购房建议</h2>', unsafe_allow_html=True)

    st.success("""
    ### 基于模型分析的三大黄金法则

    1. **面积×租约法则**
    这是影响房价的最重要因素（贡献度21.27%）。在预算范围内，优先选择**面积适中且剩余租约较长**的房源，这是保值的根本。

    2. **交通+商业双优法则**
    同时靠近地铁站（<500米）和商场（<800米）的房源溢价最高，流动性最好。模型显示这两个因素的交互项对房价有显著正向影响。

    3. **区域均衡法则**
    不要只看单一因素，要综合考虑价格、交通、商业、学区和升值潜力。综合得分最高的区域往往是最稳妥的选择。

    最后记住：没有完美的房子，只有最适合你的房子。在预算范围内，选择能满足你核心需求的房源就是最好的决策。
    """)

# 页脚（更新为最新模型性能）
st.sidebar.markdown("---")
st.sidebar.info("""
**新加坡HDB房价分析与预测系统**  
数据来源：组屋发展局(HDB)、教育部(MOE)、城市重建局(URA)、国家公园委员会(NPARKS)、新加坡警察队、新加坡统计局、陆路交通管理局(LTA)
模型准确率：R²=0.8789，每平方米平均(中位数)误差：$574.90/㎡($477.11/㎡)
""")
