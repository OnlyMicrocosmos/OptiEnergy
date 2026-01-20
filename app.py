import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 导入你的后台模型
import utils
import model_operation
import model_capacity
import model_dro
import ccg_main

# ==================== 全局配置与样式美化 ====================
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 【关键修改 1】：必须先设置绘图风格，否则它会覆盖掉后面的字体设置！
plt.style.use('seaborn-v0_8-paper')

# 【关键修改 2】：自动加载本地字体并动态获取字体名称
font_path = 'SimHei.ttf'
font_name = 'SimHei' # 默认值

if os.path.exists(font_path):
    # 将字体添加到 matplotlib 字体管理器
    fm.fontManager.addfont(font_path)
    # 动态获取注册后的字体内部名称（防止 SimHei.ttf 的内部名称不叫 'SimHei'）
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    # 设置全局字体
    plt.rcParams['font.family'] = font_name
else:
    # 本地调试时的备选
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 打印日志确认字体是否加载（调试用，可在终端查看）
print(f"当前使用的字体文件: {font_path}, 注册名称: {font_name}")

# 2. 页面基础配置
st.set_page_config(
    page_title="综合能源系统优化平台",
    layout="wide",
    page_icon="⚡"
)

# 3. 自定义 CSS (科技感、清灵风格)
st.markdown("""
<style>
    /* 全局背景：极淡的科技蓝灰渐变 */
    .stApp {
        background: linear-gradient(to bottom right, #f4f7f9, #e6eef5);
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        border-right: 1px solid #e0e0e0;
    }

    /* 标题样式 */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #1e88e5, #00acc1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }

    /* 卡片/容器样式 - 增加轻微阴影和圆角 */
    div.stButton > button {
        background: linear-gradient(90deg, #1e88e5 0%, #42a5f5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(30, 136, 229, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(30, 136, 229, 0.3);
        background: linear-gradient(90deg, #1976d2 0%, #1e88e5 100%);
    }

    /* Metric 指标卡片样式 */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.6);
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        backdrop-filter: blur(10px);
    }
    label[data-testid="stMetricLabel"] {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #2c3e50;
        font-weight: 700;
    }

    /* 表格样式 */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
st.sidebar.title("🌞 综合能源系统")
# 已删除 "基于 CVXPY..." 的 st.sidebar.info

# 功能选择 (已修改选项名称，去掉括号)
module = st.sidebar.radio(
    "功能导航",
    ["📊 基础数据预览", "⚡ 确定性运行优化", "🏗️ 容量规划", "🛡️ 鲁棒优化", "🎲 分布鲁棒"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2023 Intelligent Energy Lab")

# ==================== 1. 基础数据预览 ====================
if module == "📊 基础数据预览":
    st.title("📊 基础数据预览")

    st.markdown("##### 可视化展示系统的负荷需求、电价波动以及可再生能源出力预测。")
    st.markdown("---")

    # 加载数据
    data = utils.load_default_data()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🍃 风光资源预测")
        chart_data = pd.DataFrame({
            "光伏 (MW)": data['P_pv'],
            "风电 (MW)": data['P_wd']
        })
        st.line_chart(chart_data, color=["#FFA500", "#1E90FF"])  # 指定颜色：橙色光伏，蓝色风电

    with col2:
        st.subheader("📉 负荷与电价")
        # 创建一个更美观的 Matplotlib 图
        fig, ax1 = plt.subplots(figsize=(8, 4))

        # 设置背景透明以便融合
        fig.patch.set_alpha(0)
        ax1.patch.set_alpha(0)

        # 绘制负荷
        line1 = ax1.plot(data['L_tt'], color='#2c3e50', linewidth=2, label='负荷 (MW)')
        ax1.set_ylabel('负荷 (MW)', color='#2c3e50', fontsize=10)
        ax1.set_xlabel('时间 (h)', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='#2c3e50')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 绘制电价
        ax2 = ax1.twinx()
        line2 = ax2.plot(data['price'], color='#e74c3c', linestyle='--', linewidth=2, label='电价 (元/MWh)')
        ax2.set_ylabel('电价 (元/MWh)', color='#e74c3c', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        st.pyplot(fig)

# ==================== 2. 确定性运行优化 ====================
elif module == "⚡ 确定性运行优化":
    st.title("⚡ 确定性优化调度")
    st.markdown("##### 基于混合整数线性规划 (MILP) 的日前经济调度。")

    # 控制参数容器
    with st.container():
        st.markdown("#### ⚙️ 场景配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_pv = st.toggle("启用光伏", True)
        with col2:
            use_wind = st.toggle("启用风电", True)
        with col3:
            use_eh = st.toggle("启用电加热", True)

    st.markdown("<br>", unsafe_allow_html=True)  # 间距

    if st.button("🚀 开始运行优化"):
        with st.spinner("正在计算最优调度策略..."):
            # 准备数据
            raw_data = utils.load_default_data()
            load_data = raw_data['L_tt']
            res_data = {'pv_power': raw_data['P_pv'], 'wind_power': raw_data['P_wd']}
            params = model_operation.get_default_params()
            params['rho'] = raw_data['price']  # 更新电价
            flags = {'use_pv': use_pv, 'use_wind': use_wind, 'use_eh': use_eh}

            # 调用模型
            result = model_operation.solve_operation_model(load_data, res_data, params, flags)

            if result['status'] == 'optimal':
                st.success(f"✅ 优化成功！")

                # 关键指标卡片
                st.markdown("#### 💡 关键运行指标")
                m1, m2, m3, m4 = st.columns(4)
                stats = result['statistics']
                m1.metric("总收益 (元)", f"{result['objective_value']:,.2f}")
                m2.metric("CSP发电量", f"{stats['total_csp_generation']:.1f} MWh")
                m3.metric("新能源消纳率",
                          f"{100 - (stats['total_wind_curtail'] + stats['total_pv_curtail']) / (np.sum(raw_data['P_pv']) + np.sum(raw_data['P_wd'])) * 100:.1f} %")
                m4.metric("电加热耗电", f"{stats['total_eh_consumption']:.1f} MWh")

                st.markdown("---")

                # 绘制堆叠图
                st.subheader("📈 功率平衡堆叠图")
                df_res = pd.DataFrame({
                    "CSP出力": result['P_CSP'],
                    "风电上网": result['P_WE_WD'],
                    "光伏上网": result['P_WE_PV'],
                    "欠负荷": result['load_shed']
                })
                st.area_chart(df_res)

                # 储热状态
                st.subheader("🔋 储热系统状态 (TES)")
                st.line_chart(result['E_TES'], color="#2ecc71")

            else:
                st.error(f"❌ 求解失败: {result['message']}")

# ==================== 3. 容量规划 ====================
elif module == "🏗️ 容量规划":
    st.title("🏗️ 系统容量规划")
    st.markdown("##### 考虑全生命周期成本 (LCC) 的设备容量最优配置。")

    col_conf, col_res = st.columns([1, 2])

    with col_conf:
        with st.container():
            st.markdown("#### 🛠️ 成本参数设置")
            cost_sf = st.slider("镜场成本 ($/m²)", 50, 200, 120)
            cost_tes = st.slider("储热成本 ($/kWh)", 10, 100, 25)
            st.info("此模块将基于30年项目寿命进行优化计算。")

            if st.button("🚀 开始规划容量"):
                run_planning = True
            else:
                run_planning = False

    with col_res:
        if run_planning:
            with st.spinner("正在进行全生命周期容量优化..."):
                # 准备数据
                raw_data = utils.load_default_data()
                weather_data = {'v_wind': raw_data['v_wind'], 'I_DNI': raw_data['I_DNI']}
                cost_params = {
                    'cost_SF_per_m2': cost_sf,
                    'cost_TES_per_kWh': cost_tes,
                    'rho': raw_data['price'] / 7.0
                }

                # 调用模型
                result = model_capacity.solve_capacity_planning(
                    weather_data, raw_data['L_tt'], cost_params
                )

                if result['status'] == 'optimal':
                    st.success("✅ 规划完成！")

                    # 展示最优容量
                    opt = result['optimal_capacity']
                    st.markdown("#### 🏆 最优配置结果")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("风电容量", f"{opt['P_wind_capacity']:.2f} MW")
                    c2.metric("光伏容量", f"{opt['P_pv_capacity']:.2f} MW")
                    c3.metric("镜场面积", f"{opt['A_SF']:.0f} m²")
                    c4.metric("储热容量", f"{opt['E_TES_max']:.2f} MWh")

                    st.markdown("#### 💰 经济性分析")
                    econ = result['economics']

                    e1, e2, e3 = st.columns(3)
                    e1.metric("LCOE (度电成本)", f"${econ['LCOE_kWh']:.4f}/kWh")
                    e2.metric("初始总投资", f"${econ['total_investment'] / 1e6:.2f} M")
                    e3.metric("30年总净收益", f"${econ['total_30year_profit'] / 1e6:.2f} M")
                else:
                    st.error("❌ 规划失败，请检查参数。")
        else:
            st.info("👈 请在左侧设置参数并点击开始按钮")

# ==================== 4. 鲁棒优化 ====================
elif module == "🛡️ 鲁棒优化":
    # 标题修改：去掉括号
    st.title("🛡️ 鲁棒机组组合")
    st.markdown("##### 采用列约束生成 (C&CG) 算法处理源荷不确定性。")

    st.warning("⚠️ 算法涉及多轮主子问题迭代，计算可能需要 1-2 分钟，请耐心等待。")

    col_param, col_main = st.columns([1, 3])

    with col_param:
        st.markdown("#### 参数配置")
        max_iter = st.number_input("最大迭代次数", min_value=1, max_value=50, value=10)
        # 按钮修改：改成“运行算法”
        run_btn = st.button("🚀 运行算法")

    with col_main:
        if run_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 配置 C&CG
            config = ccg_main.CCGConfig(max_iterations=max_iter, verbose=False, plot_results=False)

            status_text.text("正在初始化算法...")

            # 调用模型
            try:
                # 确保 ccg_main.py 里有这个函数，或者根据之前的修改使用 run_ccg_algorithm
                if hasattr(ccg_main, 'run_ccg_algorithm'):
                    result = ccg_main.run_ccg_algorithm(config)
                else:
                    result = ccg_main.run_ccg_algo(config)  # 兼容旧命名

                progress_bar.progress(100)
                status_text.empty()

                if result.converged:
                    st.success(f"✅ 算法收敛！共迭代 {result.iterations} 次")
                else:
                    st.warning(f"⚠️ 达到最大迭代次数，当前 Gap: {result.gap:.2f}")

                # 收敛曲线
                st.subheader("📉 迭代收敛过程 (LB vs UB)")
                conv_df = pd.DataFrame({
                    "下界 (LB)": result.LB_history,
                    "上界 (UB)": result.UB_history
                })
                st.line_chart(conv_df, color=["#2ecc71", "#e74c3c"])

                # 最坏场景展示
                st.subheader("🌪️ 最坏场景下的负荷波动")
                if result.final_solution_details:
                    wc_load = result.final_solution_details['Load_u']
                    chart_data = pd.DataFrame({"最坏场景负荷": wc_load})
                    st.line_chart(chart_data)

            except Exception as e:
                st.error(f"❌ 运行出错: {str(e)}")
                st.write("请检查 ccg_main.py 的接口定义。")

# ==================== 5. 分布鲁棒优化 ====================
elif module == "🎲 分布鲁棒":
    # 标题修改：去掉括号
    st.title("🎲 分布鲁棒优化")
    st.markdown("##### 基于 Wasserstein 距离的分布鲁棒优化模型。")

    with st.container():
        st.markdown("#### 🎯 不确定性设置")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            rho = st.slider("不确定性半径 (Rho)", 0.0, 0.5, 0.1, 0.01)
        with c2:
            scenario = st.selectbox("天气场景偏好", ["晴天 (Sunny)", "多云 (Cloudy)", "雨天 (Rainy)"])

        scenario_map = {
            "晴天 (Sunny)": model_dro.ScenarioType.SUNNY,
            "多云 (Cloudy)": model_dro.ScenarioType.CLOUDY,
            "雨天 (Rainy)": model_dro.ScenarioType.RAINY
        }

        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            # 按钮修改：改成“运行模型”
            run_dro = st.button("🚀 运行模型")

    st.markdown("---")

    if run_dro:
        with st.spinner("正在求解分布鲁棒模型..."):
            # 准备数据
            raw_data = utils.load_default_data()
            nominal_data = {
                'v_wind': raw_data['v_wind'],
                'I_DNI': raw_data['I_DNI'],
                'price': raw_data['price'] / 7.0,  # 简单换算成美元
                'L_tt': raw_data['L_tt']
            }

            # 运行模型
            res = model_dro.solve_dro_model(
                nominal_data,
                rho=rho,
                scenario_type=scenario_map[scenario],
                verbose=False
            )

            if res.status in ['optimal', 'optimal_inaccurate']:
                st.success(f"✅ 求解成功！")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最坏情况成本", f"${res.worst_case_cost:.2f}")
                with col2:
                    st.metric("总利润", f"${res.total_profit:,.2f}")
                with col3:
                    st.metric("LCOE", f"${res.LCOE:.4f}/kWh")

                # 展示鲁棒容量规划结果
                st.subheader("🛠️ 鲁棒容量配置建议")

                # 使用自定义样式的列
                res_df = pd.DataFrame({
                    "项目": ["风电容量 (MW)", "光伏容量 (MW)", "储热容量 (MWh)", "电加热容量 (MW)", "镜场面积 (m²)"],
                    "推荐配置": [
                        f"{res.P_wind_capacity:.2f}",
                        f"{res.P_pv_capacity:.2f}",
                        f"{res.E_TES_max:.2f}",
                        f"{res.P_EH_Max:.2f}",
                        f"{res.A_SF:.0f}"
                    ]
                })
                st.dataframe(res_df, use_container_width=True)

            else:
                st.error(f"❌ 求解失败: {res.status}")
