"""
能源系统容量规划优化模型 - 基于CVXPY实现
优化变量：风电容量、光伏容量、镜场面积、储热容量、电加热容量
目标：最大化30年净收益
"""

import cvxpy as cp
import numpy as np
from typing import Dict, Any, Optional, Tuple


def solve_capacity_planning(
        weather_data: Dict[str, np.ndarray],
        load_data: np.ndarray,
        cost_params: Dict[str, float],
        system_params: Optional[Dict[str, float]] = None,
        capacity_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    能源系统容量规划优化模型

    优化变量：
    - 风电装机容量 (MW)
    - 光伏装机容量 (MW)
    - 太阳场镜场面积 (m²)
    - 储热系统容量 (MWh)
    - 电加热器容量 (MW)

    Parameters:
    -----------
    weather_data : dict
        气象数据，包含:
        - 'v_wind': 风速数组 (24,) (m/s)
        - 'I_DNI': 太阳直射辐射数组 (24,) (W/m²)

    load_data : np.ndarray
        24小时负荷数据 (MW)

    cost_params : dict
        成本参数，包含:
        - 'cost_SF_per_m2': 镜场单位成本 ($/m²)
        - 'cost_TES_per_kWh': 储热单位成本 ($/kWh)
        - 'cost_PB_per_kW': 发电机组单位成本 ($/kW)
        - 'cost_PV_per_kW': 光伏单位成本 ($/kW)
        - 'cost_wind_per_kW': 风电单位成本 ($/kW)
        - 'cost_EH_per_kW': 电加热器单位成本 ($/kW)
        - 'OM_CSP_per_kW': CSP年运维成本 ($/kW/年)
        - 'OM_PV_per_kW': 光伏年运维成本 ($/kW/年)
        - 'OM_wind_per_kW': 风电年运维成本 ($/kW/年)
        - 'OM_SF_per_m2': 镜场年运维成本 ($/m²/年)
        - 'rho': 电价数组 (24,) ($/MWh)

    system_params : dict, optional
        系统技术参数

    capacity_bounds : dict, optional
        容量边界约束

    Returns:
    --------
    dict : 优化结果，包含最优容量配置、经济指标等
    """

    # ==================== 1. 参数提取与默认值设置 ====================
    T = 24  # 时间段数
    Delta_t = 1  # 时间间隔（小时）
    n_years = 30  # 项目寿命
    discount_rate = 0.05  # 折现率
    days_per_year = 365  # 年运行天数

    # 气象数据
    v_wind = np.array(weather_data.get('v_wind', np.ones(T) * 7)).flatten()
    I_DNI = np.array(weather_data.get('I_DNI', np.zeros(T))).flatten()

    # 负荷数据
    L_tt = np.array(load_data).flatten()

    # 电价数据
    rho = np.array(cost_params.get('rho', np.ones(T) * 100)).flatten()

    # 成本参数 - 使用默认值（基于first.txt）
    cost_SF_per_m2 = cost_params.get('cost_SF_per_m2', 120)  # $/m²
    cost_TES_per_kWh = cost_params.get('cost_TES_per_kWh', 25)  # $/kWh
    cost_PB_per_kW = cost_params.get('cost_PB_per_kW', 880)  # $/kW
    cost_PV_per_kW = cost_params.get('cost_PV_per_kW', 790)  # $/kW
    cost_wind_per_kW = cost_params.get('cost_wind_per_kW', 1180)  # $/kW
    cost_EH_per_kW = cost_params.get('cost_EH_per_kW', 40)  # $/kW

    OM_CSP_per_kW = cost_params.get('OM_CSP_per_kW', 24)  # $/kW/年
    OM_PV_per_kW = cost_params.get('OM_PV_per_kW', 14)  # $/kW/年
    OM_wind_per_kW = cost_params.get('OM_wind_per_kW', 50)  # $/kW/年
    OM_SF_per_m2 = cost_params.get('OM_SF_per_m2', 7.82)  # $/m²/年

    # 系统技术参数
    if system_params is None:
        system_params = {}

    # 风力发电参数
    v_ci = system_params.get('v_ci', 2.3)  # 切入风速 (m/s)
    v_r = system_params.get('v_r', 10)  # 额定风速 (m/s)
    v_co = system_params.get('v_co', 20)  # 切出风速 (m/s)
    P_r_single = system_params.get('P_r_single', 2.4)  # 单台风机额定功率 (MW)

    # 光伏发电参数
    eta_pv = system_params.get('eta_pv', 0.167)  # 光伏效率
    S_pv_i = system_params.get('S_pv_i', 2.3)  # 单个光伏组件面积 (m²)
    D_ref = system_params.get('D_ref', 800)  # 参考辐射强度 (W/m²)
    T_ref = system_params.get('T_ref', 20)  # 参考温度 (℃)
    T_a = system_params.get('T_a', 25)  # 环境温度 (℃)
    N_CT = system_params.get('N_CT', 45)  # 额定运行温度 (℃)

    # CSP系统参数
    eta_SF = system_params.get('eta_SF', 0.38)  # 光热转化效率
    gamma_TES = system_params.get('gamma_TES', 0.038)  # 储热损失系数
    E_TES_min = system_params.get('E_TES_min', 20)  # 最小储热能量 (MWh)
    E_init = system_params.get('E_init', 80)  # 初始储热能量 (MWh)
    eta_TES_cha = system_params.get('eta_TES_cha', 0.98)
    eta_TES_dis = system_params.get('eta_TES_dis', 0.98)
    Q_HTF_TES_max = system_params.get('Q_HTF_TES_max', 150)
    Q_TES_HTF_max = system_params.get('Q_TES_HTF_max', 150)
    eta_PB = system_params.get('eta_PB', 0.37)
    P_PB_min = system_params.get('P_PB_min', 0)
    P_PB_max = system_params.get('P_PB_max', 80)
    Delta_P_Ru_PB = system_params.get('Delta_P_Ru_PB', 50)
    Delta_P_Rd_PB = system_params.get('Delta_P_Rd_PB', 50)
    eta_EH = system_params.get('eta_EH', 0.95)

    # 运行约束参数
    curtail_absorption_rate = system_params.get('curtail_absorption_rate', 0.9)  # 弃能吸纳率下限
    max_load_shed_rate = system_params.get('max_load_shed_rate', 0.1)  # 最大切负荷比例

    # 容量边界约束
    if capacity_bounds is None:
        capacity_bounds = {}

    A_SF_min = capacity_bounds.get('A_SF_min', 5000)
    A_SF_max = capacity_bounds.get('A_SF_max', 5e6)
    E_TES_max_min = capacity_bounds.get('E_TES_max_min', 0)
    E_TES_max_max = capacity_bounds.get('E_TES_max_max', 3000)
    P_EH_max_min = capacity_bounds.get('P_EH_max_min', 0)
    P_EH_max_max = capacity_bounds.get('P_EH_max_max', 200)
    P_wind_cap_min = capacity_bounds.get('P_wind_cap_min', 0)
    P_wind_cap_max = capacity_bounds.get('P_wind_cap_max', 500)
    P_pv_cap_min = capacity_bounds.get('P_pv_cap_min', 0)
    P_pv_cap_max = capacity_bounds.get('P_pv_cap_max', 400)

    # ==================== 2. 预计算风电和光伏单位出力 ====================

    # 计算单台风机各时段出力 (MW)
    P_w_i = np.zeros(T)
    for t in range(T):
        v = v_wind[t]
        if v < v_ci or v >= v_co:
            P_w_i[t] = 0
        elif v >= v_ci and v < v_r:
            P_w_i[t] = P_r_single * (v ** 3 - v_ci ** 3) / (v_r ** 3 - v_ci ** 3)
        else:  # v >= v_r and v < v_co
            P_w_i[t] = P_r_single

    # 计算单位光伏容量各时段出力系数
    P_pv_unit = np.zeros(T)
    P_pv_single_STC = eta_pv * S_pv_i * D_ref / 1e6  # 单个组件STC功率 (MW)

    for t in range(T):
        # 计算光伏组件运行温度
        T_pv = T_a + (N_CT - T_ref) * (I_DNI[t] / D_ref) if D_ref > 0 else T_a
        # 计算单位光伏功率
        P_pv_unit[t] = eta_pv * S_pv_i * I_DNI[t] / 1e6 * (1 - 0.005 * (T_pv - 25))
        if P_pv_unit[t] < 0:
            P_pv_unit[t] = 0

    # 计算单位容量的出力系数
    # 风电：P_wd = (P_wind_capacity / P_r_single) * P_w_i
    # 光伏：P_pv = (P_pv_capacity / P_pv_single_STC) * P_pv_unit

    wind_capacity_factor = P_w_i / P_r_single if P_r_single > 0 else np.zeros(T)
    pv_capacity_factor = P_pv_unit / P_pv_single_STC if P_pv_single_STC > 0 else np.zeros(T)

    # ==================== 3. 定义决策变量 ====================

    # --- 容量规划变量（优化变量）---
    P_wind_capacity = cp.Variable(1, nonneg=True)  # 风电装机容量 (MW)
    P_pv_capacity = cp.Variable(1, nonneg=True)  # 光伏装机容量 (MW)
    A_SF = cp.Variable(1, nonneg=True)  # 镜场面积 (m²)
    E_TES_max = cp.Variable(1, nonneg=True)  # 储热容量 (MWh)
    P_EH_max = cp.Variable(1, nonneg=True)  # 电加热容量 (MW)

    # --- 运行变量 ---
    # CSP系统
    Q_SF_in = cp.Variable(T, nonneg=True)
    Q_SF_HTF = cp.Variable(T, nonneg=True)
    Q_SF_cur = cp.Variable(T, nonneg=True)
    Q_Loss = cp.Variable(T, nonneg=True)

    # 储热系统
    Q_TES_HTF = cp.Variable(T, nonneg=True)
    Q_HTF_TES = cp.Variable(T, nonneg=True)
    Q_TES_cha = cp.Variable(T, nonneg=True)
    Q_TES_dis = cp.Variable(T, nonneg=True)
    E_TES = cp.Variable(T, nonneg=True)

    # 发电机组
    Q_HTF_PB = cp.Variable(T, nonneg=True)
    P_PB = cp.Variable(T, nonneg=True)
    P_CSP = cp.Variable(T, nonneg=True)

    # 二值变量
    x_PB = cp.Variable(T, boolean=True)
    x_TES_cha = cp.Variable(T, boolean=True)
    x_TES_dis = cp.Variable(T, boolean=True)

    # 电加热器
    Q_EH_HTF = cp.Variable(T, nonneg=True)
    P_EH = cp.Variable(T, nonneg=True)

    # 风电变量
    P_WE_WD = cp.Variable(T, nonneg=True)
    P_WC_WD = cp.Variable(T, nonneg=True)

    # 光伏变量
    P_WE_PV = cp.Variable(T, nonneg=True)
    P_WC_PV = cp.Variable(T, nonneg=True)

    # 负荷变量
    load_shed = cp.Variable(T, nonneg=True)

    # ==================== 4. 构造约束 ====================
    constraints = []

    # --- 4.1 容量边界约束 ---
    constraints.append(A_SF >= A_SF_min)
    constraints.append(A_SF <= A_SF_max)
    constraints.append(E_TES_max >= E_TES_max_min)
    constraints.append(E_TES_max <= E_TES_max_max)
    constraints.append(P_EH_max >= P_EH_max_min)
    constraints.append(P_EH_max <= P_EH_max_max)
    constraints.append(P_wind_capacity >= P_wind_cap_min)
    constraints.append(P_wind_capacity <= P_wind_cap_max)
    constraints.append(P_pv_capacity >= P_pv_cap_min)
    constraints.append(P_pv_capacity <= P_pv_cap_max)

    # --- 4.2 运行约束（每个时段）---
    for t in range(T):
        # HTF能量平衡
        constraints.append(
            Q_SF_HTF[t] + Q_TES_HTF[t] + Q_EH_HTF[t] == Q_HTF_TES[t] + Q_HTF_PB[t]
        )

        # 太阳场约束
        # Q_SF_in = (eta_SF * I_DNI * A_SF) / 1e6 - Q_Loss
        constraints.append(
            Q_SF_in[t] == (eta_SF * I_DNI[t] / 1e6) * A_SF - Q_Loss[t]
        )
        constraints.append(Q_SF_HTF[t] == Q_SF_in[t] - Q_SF_cur[t])

        # 风电功率守恒（与容量变量相关）
        # P_wd(t) = wind_capacity_factor(t) * P_wind_capacity
        P_wd_t = wind_capacity_factor[t] * P_wind_capacity
        constraints.append(P_WE_WD[t] + P_WC_WD[t] == P_wd_t)

        # 光伏功率守恒（与容量变量相关）
        # P_pv(t) = pv_capacity_factor(t) * P_pv_capacity
        P_pv_t = pv_capacity_factor[t] * P_pv_capacity
        constraints.append(P_WE_PV[t] + P_WC_PV[t] == P_pv_t)

        # 电加热只能使用弃风弃光
        constraints.append(P_EH[t] <= P_WC_WD[t] + P_WC_PV[t])

        # 负荷平衡
        constraints.append(
            P_CSP[t] + P_WE_WD[t] + P_WE_PV[t] + load_shed[t] == L_tt[t]
        )
        constraints.append(load_shed[t] <= L_tt[t])

        # 储热系统关系
        constraints.append(Q_TES_cha[t] == eta_TES_cha * Q_HTF_TES[t])
        constraints.append(Q_TES_dis[t] == Q_TES_HTF[t] / eta_TES_dis)

        # 储热能量动态
        if t == 0:
            # 初始时刻约束：E_TES(0) = E_init，但E_init不能超过E_TES_max
            # 使用一个合理的初始值
            constraints.append(E_TES[t] <= E_TES_max)
            constraints.append(E_TES[t] >= E_TES_min)
        else:
            constraints.append(
                E_TES[t] == (1 - gamma_TES) * E_TES[t - 1] +
                (Q_TES_cha[t] - Q_TES_dis[t]) * Delta_t
            )

        # 储热能量上下限
        constraints.append(E_TES[t] >= E_TES_min)
        constraints.append(E_TES[t] <= E_TES_max)

        # 发电机组热电转换
        constraints.append(Q_HTF_PB[t] == P_PB[t] / eta_PB)

        # 发电机组出力约束
        constraints.append(P_PB[t] >= P_PB_min * x_PB[t])
        constraints.append(P_PB[t] <= P_PB_max * x_PB[t])

        # 爬坡约束
        if t >= 1:
            constraints.append(P_PB[t] - P_PB[t - 1] <= Delta_P_Ru_PB)
            constraints.append(P_PB[t - 1] - P_PB[t] <= Delta_P_Rd_PB)

        # CSP净发电
        constraints.append(P_CSP[t] == P_PB[t])

        # 电加热约束
        constraints.append(Q_EH_HTF[t] == eta_EH * P_EH[t])
        constraints.append(Q_EH_HTF[t] <= Q_HTF_TES[t] + Q_HTF_PB[t])
        constraints.append(P_EH[t] <= P_EH_max)

        # 充放热互斥
        constraints.append(x_TES_cha[t] + x_TES_dis[t] <= 1)

        # 储热功率限制
        constraints.append(Q_HTF_TES[t] <= Q_HTF_TES_max * x_TES_cha[t])
        constraints.append(Q_TES_HTF[t] <= Q_TES_HTF_max * x_TES_dis[t])

    # --- 4.3 全局约束 ---
    # 弃风弃光吸纳率约束：电加热消纳量 >= 90% * 总弃能量
    total_curtail = cp.sum(P_WC_WD) + cp.sum(P_WC_PV)
    total_eh = cp.sum(P_EH)
    constraints.append(total_eh >= curtail_absorption_rate * total_curtail)

    # 切负荷上限：总切负荷 <= 10% * 总负荷
    constraints.append(cp.sum(load_shed) <= max_load_shed_rate * np.sum(L_tt))

    # 初始状态
    constraints.append(x_PB[0] == 1)

    # ==================== 5. 成本计算 ====================

    # 投资成本（美元）
    cost_SF = cost_SF_per_m2 * A_SF  # 镜场
    cost_TES = cost_TES_per_kWh * 1000 * E_TES_max  # 储热（MWh转kWh）
    cost_PB = cost_PB_per_kW * 1000 * P_PB_max  # 发电机组（固定）
    cost_PV = cost_PV_per_kW * 1000 * P_pv_capacity  # 光伏
    cost_wind = cost_wind_per_kW * 1000 * P_wind_capacity  # 风电
    cost_EH = cost_EH_per_kW * 1000 * P_EH_max  # 电加热

    cost_investment = cost_SF + cost_TES + cost_PB + cost_PV + cost_wind + cost_EH

    # 年运维成本（美元/年）
    OM_CSP = OM_CSP_per_kW * 1000 * P_PB_max  # CSP运维（固定）
    OM_PV = OM_PV_per_kW * 1000 * P_pv_capacity  # 光伏运维
    OM_wind = OM_wind_per_kW * 1000 * P_wind_capacity  # 风电运维
    OM_SF = OM_SF_per_m2 * A_SF  # 镜场运维

    cost_yearly_OM = OM_CSP + OM_PV + OM_wind + OM_SF

    # 30年总成本
    total_cost = cost_investment + cost_yearly_OM * n_years

    # ==================== 6. 收益计算 ====================

    # 单日售电收入
    daily_revenue = cp.sum(cp.multiply(rho, P_PB + P_WE_WD + P_WE_PV))

    # 30年总收入
    total_revenue = daily_revenue * days_per_year * n_years

    # ==================== 7. 目标函数 ====================
    # 最大化30年净收益 = 总收入 - 总成本
    objective = total_revenue - total_cost

    # ==================== 8. 求解 ====================
    problem = cp.Problem(cp.Maximize(objective), constraints)

    # 选择求解器
    solvers_to_try = []
    if 'CBC' in cp.installed_solvers():
        solvers_to_try.append(cp.CBC)
    if 'SCIP' in cp.installed_solvers():
        solvers_to_try.append(cp.SCIP)
    if 'GLPK_MI' in cp.installed_solvers():
        solvers_to_try.append(cp.GLPK_MI)
    if 'HIGHS' in cp.installed_solvers():
        solvers_to_try.append(cp.HIGHS)

    if not solvers_to_try:
        solvers_to_try = [None]

    solve_success = False
    solve_info = ""

    for solver in solvers_to_try:
        try:
            if solver is not None:
                problem.solve(solver=solver, verbose=False)
            else:
                problem.solve(verbose=False)

            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                solve_success = True
                solve_info = f"求解成功，使用求解器: {solver if solver else 'default'}"
                break
        except Exception as e:
            solve_info = f"求解器 {solver} 失败: {str(e)}"
            continue

    # ==================== 9. 结果提取与LCOE计算 ====================
    if solve_success:
        # 提取容量优化结果
        P_wind_cap_opt = float(P_wind_capacity.value[0]) if P_wind_capacity.value is not None else 0
        P_pv_cap_opt = float(P_pv_capacity.value[0]) if P_pv_capacity.value is not None else 0
        A_SF_opt = float(A_SF.value[0]) if A_SF.value is not None else 0
        E_TES_max_opt = float(E_TES_max.value[0]) if E_TES_max.value is not None else 0
        P_EH_max_opt = float(P_EH_max.value[0]) if P_EH_max.value is not None else 0

        # 提取运行结果
        P_CSP_opt = P_CSP.value if P_CSP.value is not None else np.zeros(T)
        P_WE_WD_opt = P_WE_WD.value if P_WE_WD.value is not None else np.zeros(T)
        P_WE_PV_opt = P_WE_PV.value if P_WE_PV.value is not None else np.zeros(T)
        P_WC_WD_opt = P_WC_WD.value if P_WC_WD.value is not None else np.zeros(T)
        P_WC_PV_opt = P_WC_PV.value if P_WC_PV.value is not None else np.zeros(T)
        P_EH_opt = P_EH.value if P_EH.value is not None else np.zeros(T)
        E_TES_opt = E_TES.value if E_TES.value is not None else np.zeros(T)
        load_shed_opt = load_shed.value if load_shed.value is not None else np.zeros(T)

        # 计算发电量统计
        daily_gen_csp = np.sum(P_CSP_opt) * Delta_t
        daily_gen_wind = np.sum(P_WE_WD_opt) * Delta_t
        daily_gen_pv = np.sum(P_WE_PV_opt) * Delta_t
        daily_gen_total = daily_gen_csp + daily_gen_wind + daily_gen_pv

        annual_gen_csp = daily_gen_csp * days_per_year
        annual_gen_wind = daily_gen_wind * days_per_year
        annual_gen_pv = daily_gen_pv * days_per_year
        annual_gen_total = daily_gen_total * days_per_year

        # 弃能统计
        daily_curtail_wind = np.sum(P_WC_WD_opt) * Delta_t
        daily_curtail_pv = np.sum(P_WC_PV_opt) * Delta_t
        daily_eh_consumption = np.sum(P_EH_opt) * Delta_t

        annual_curtail_wind = daily_curtail_wind * days_per_year
        annual_curtail_pv = daily_curtail_pv * days_per_year
        annual_eh_consumption = daily_eh_consumption * days_per_year
        actual_annual_curtail = annual_curtail_wind + annual_curtail_pv - annual_eh_consumption

        # 吸纳率
        total_curtail_val = daily_curtail_wind + daily_curtail_pv
        absorption_rate = (daily_eh_consumption / total_curtail_val * 100) if total_curtail_val > 0 else 100

        # 提取数值 (使用 np.array(...).item() 确保将数组转换为标量)
        def to_scalar(val):
            if val is None: return 0.0
            return np.array(val).item()

        cost_inv_val = to_scalar(cost_investment.value)
        cost_om_val = to_scalar(cost_yearly_OM.value)
        total_cost_val = to_scalar(total_cost.value)

        # ==================== LCOE计算 ====================
        # LCOE = (投资成本 + 运维成本现值) / 发电量现值

        # 计算折现因子
        def calc_present_value_factor(rate, years):
            """计算年金现值因子"""
            if rate == 0:
                return years
            return (1 - (1 + rate) ** (-years)) / rate

        pv_factor = calc_present_value_factor(discount_rate, n_years)

        # 运维成本现值
        om_present_value = cost_om_val * pv_factor

        # 总成本现值
        total_cost_present = cost_inv_val + om_present_value

        # 发电量现值
        gen_present_value = annual_gen_total * pv_factor

        # LCOE ($/MWh)
        LCOE_MWh = total_cost_present / gen_present_value if gen_present_value > 0 else float('inf')
        LCOE_kWh = LCOE_MWh / 1000  # $/kWh

        # 总收益
        total_profit = problem.value

        results = {
            'status': 'optimal',
            'message': solve_info,

            # 容量优化结果
            'optimal_capacity': {
                'P_wind_capacity': P_wind_cap_opt,  # MW
                'P_pv_capacity': P_pv_cap_opt,  # MW
                'A_SF': A_SF_opt,  # m²
                'E_TES_max': E_TES_max_opt,  # MWh
                'P_EH_max': P_EH_max_opt,  # MW
            },

            # 经济指标
            'economics': {
                'total_investment': cost_inv_val,  # $
                'annual_OM_cost': cost_om_val,  # $/年
                'total_30year_cost': total_cost_val,  # $
                'total_30year_profit': total_profit,  # $
                'LCOE_MWh': LCOE_MWh,  # $/MWh
                'LCOE_kWh': LCOE_kWh,  # $/kWh
            },

            # 发电量统计
            'generation': {
                'daily_csp': daily_gen_csp,  # MWh/日
                'daily_wind': daily_gen_wind,  # MWh/日
                'daily_pv': daily_gen_pv,  # MWh/日
                'daily_total': daily_gen_total,  # MWh/日
                'annual_csp': annual_gen_csp,  # MWh/年
                'annual_wind': annual_gen_wind,  # MWh/年
                'annual_pv': annual_gen_pv,  # MWh/年
                'annual_total': annual_gen_total,  # MWh/年
            },

            # 弃能统计
            'curtailment': {
                'annual_wind_curtail': annual_curtail_wind,  # MWh/年
                'annual_pv_curtail': annual_curtail_pv,  # MWh/年
                'annual_eh_absorption': annual_eh_consumption,  # MWh/年
                'actual_annual_curtail': actual_annual_curtail,  # MWh/年
                'absorption_rate': absorption_rate,  # %
            },

            # 运行曲线
            'operation': {
                'P_CSP': P_CSP_opt,
                'P_WE_WD': P_WE_WD_opt,
                'P_WE_PV': P_WE_PV_opt,
                'P_WC_WD': P_WC_WD_opt,
                'P_WC_PV': P_WC_PV_opt,
                'P_EH': P_EH_opt,
                'E_TES': E_TES_opt,
                'load_shed': load_shed_opt,
            }
        }
    else:
        results = {
            'status': 'failed',
            'message': f"求解失败: {solve_info}",
            'optimal_capacity': None,
            'economics': None,
            'generation': None,
            'curtailment': None,
            'operation': None,
        }

    return results


def get_default_cost_params() -> Dict[str, Any]:
    """
    获取默认成本参数（基于first.txt）

    Returns:
    --------
    dict : 默认成本参数
    """
    return {
        # 投资成本
        'cost_SF_per_m2': 120,  # 镜场 ($/m²)
        'cost_TES_per_kWh': 25,  # 储热 ($/kWh)
        'cost_PB_per_kW': 880,  # 发电机组 ($/kW)
        'cost_PV_per_kW': 790,  # 光伏 ($/kW)
        'cost_wind_per_kW': 1180,  # 风电 ($/kW)
        'cost_EH_per_kW': 40,  # 电加热 ($/kW)

        # 年运维成本
        'OM_CSP_per_kW': 24,  # CSP ($/kW/年)
        'OM_PV_per_kW': 14,  # 光伏 ($/kW/年)
        'OM_wind_per_kW': 50,  # 风电 ($/kW/年)
        'OM_SF_per_m2': 7.82,  # 镜场 ($/m²/年)

        # 电价
        'rho': np.array([45, 45, 45, 45, 45, 45, 45, 100, 100, 185,
                         185, 185, 185, 185, 100, 100, 185, 185, 185,
                         185, 185, 185, 45, 45]),
    }


def get_default_weather_data() -> Dict[str, np.ndarray]:
    """
    获取默认气象数据（基于first.txt）

    Returns:
    --------
    dict : 默认气象数据
    """
    return {
        'v_wind': np.array([6.47, 6.7, 6.57, 7.53, 7.33, 7.22, 7.4, 7.1,
                            6.82, 7.41, 6.07, 6.16, 6.84, 7.01, 6.77, 7.83,
                            7.32, 7.86, 7.53, 7.29, 6.59, 7.08, 6.61, 6.48]),

        'I_DNI': np.array([0, 0, 0, 0, 0, 50, 200, 350, 500, 750,
                           900, 1000, 1050, 950, 800, 700, 450, 200,
                           50, 0, 0, 0, 0, 0]),
    }


def print_capacity_planning_results(results: Dict[str, Any]) -> None:
    """
    打印容量规划优化结果

    Parameters:
    -----------
    results : dict
        优化结果字典
    """
    if results['status'] != 'optimal':
        print(f"优化失败: {results['message']}")
        return

    print("\n" + "=" * 70)
    print("                    容量规划优化结果")
    print("=" * 70)

    # 容量配置结果
    cap = results['optimal_capacity']
    print("\n【最优容量配置】")
    print(f"{'风电装机容量:':<25} {cap['P_wind_capacity']:>15.2f} MW")
    print(f"{'光伏装机容量:':<25} {cap['P_pv_capacity']:>15.2f} MW")
    print(f"{'镜场面积:':<25} {cap['A_SF']:>15.2f} m²")
    print(f"{'储热容量:':<25} {cap['E_TES_max']:>15.2f} MWh")
    print(f"{'电加热器容量:':<25} {cap['P_EH_max']:>15.2f} MW")

    # 经济指标
    eco = results['economics']
    print("\n【经济性指标】")
    print(f"{'总投资成本:':<25} {eco['total_investment'] / 1e6:>15.3f} 百万美元")
    print(f"{'年运维成本:':<25} {eco['annual_OM_cost'] / 1e6:>15.3f} 百万美元/年")
    print(f"{'30年总成本:':<25} {eco['total_30year_cost'] / 1e6:>15.3f} 百万美元")
    print(f"{'30年总净收益:':<25} {eco['total_30year_profit'] / 1e6:>15.3f} 百万美元")
    print(f"{'LCOE:':<25} {eco['LCOE_kWh']:>15.6f} $/kWh")
    print(f"{'LCOE:':<25} {eco['LCOE_MWh']:>15.3f} $/MWh")

    # 发电量统计
    gen = results['generation']
    print("\n【发电量统计】")
    print(f"{'年光热发电量:':<25} {gen['annual_csp'] / 1e3:>15.3f} GWh/年")
    print(f"{'年风电发电量:':<25} {gen['annual_wind'] / 1e3:>15.3f} GWh/年")
    print(f"{'年光伏发电量:':<25} {gen['annual_pv'] / 1e3:>15.3f} GWh/年")
    print(f"{'年总发电量:':<25} {gen['annual_total'] / 1e3:>15.3f} GWh/年")

    # 弃能统计
    cur = results['curtailment']
    print("\n【弃能统计】")
    print(f"{'年弃风量:':<25} {cur['annual_wind_curtail'] / 1e3:>15.3f} GWh/年")
    print(f"{'年弃光量:':<25} {cur['annual_pv_curtail'] / 1e3:>15.3f} GWh/年")
    print(f"{'电加热吸纳量:':<25} {cur['annual_eh_absorption'] / 1e3:>15.3f} GWh/年")
    print(f"{'实际弃能量:':<25} {cur['actual_annual_curtail'] / 1e3:>15.3f} GWh/年")
    print(f"{'弃能吸纳率:':<25} {cur['absorption_rate']:>14.2f} %")

    print("\n" + "=" * 70)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试：复现first.txt的场景

    # 负荷数据
    load_data = np.array([23, 20, 32, 35, 40, 51, 56, 70, 64, 50,
                          45, 40, 49, 38, 42, 50, 71, 80, 84, 71,
                          60, 42, 24, 20])

    # 气象数据
    weather_data = get_default_weather_data()

    # 成本参数
    cost_params = get_default_cost_params()

    # 系统参数（可选）
    system_params = {
        'v_ci': 2.3,
        'v_r': 10,
        'v_co': 20,
        'P_r_single': 2.4,
        'eta_pv': 0.167,
        'S_pv_i': 2.3,
        'D_ref': 800,
        'eta_SF': 0.38,
        'gamma_TES': 0.038,
        'E_TES_min': 20,
        'E_init': 80,
        'eta_TES_cha': 0.98,
        'eta_TES_dis': 0.98,
        'Q_HTF_TES_max': 150,
        'Q_TES_HTF_max': 150,
        'eta_PB': 0.37,
        'P_PB_min': 0,
        'P_PB_max': 80,
        'Delta_P_Ru_PB': 50,
        'Delta_P_Rd_PB': 50,
        'eta_EH': 0.95,
        'curtail_absorption_rate': 0.9,
        'max_load_shed_rate': 0.1,
    }

    # 容量边界
    capacity_bounds = {
        'A_SF_min': 5000,
        'A_SF_max': 5e6,
        'E_TES_max_min': 0,
        'E_TES_max_max': 3000,
        'P_EH_max_min': 0,
        'P_EH_max_max': 200,
        'P_wind_cap_min': 0,
        'P_wind_cap_max': 500,
        'P_pv_cap_min': 0,
        'P_pv_cap_max': 400,
    }

    print("开始容量规划优化...")
    print(f"可用求解器: {cp.installed_solvers()}")

    # 求解
    results = solve_capacity_planning(
        weather_data=weather_data,
        load_data=load_data,
        cost_params=cost_params,
        system_params=system_params,
        capacity_bounds=capacity_bounds
    )

    # 打印结果
    print_capacity_planning_results(results)

    # 如果优化成功，绘制运行曲线
    if results['status'] == 'optimal':
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            time = np.arange(1, 25)
            op = results['operation']

            # 图1：发电出力
            ax1 = axes[0]
            ax1.bar(time - 0.2, op['P_CSP'], width=0.2, label='CSP', alpha=0.8)
            ax1.bar(time, op['P_WE_WD'], width=0.2, label='Wind', alpha=0.8)
            ax1.bar(time + 0.2, op['P_WE_PV'], width=0.2, label='PV', alpha=0.8)
            ax1.plot(time, load_data, 'k--', linewidth=2, label='Load')
            ax1.set_xlabel('Hour')
            ax1.set_ylabel('Power (MW)')
            ax1.set_title('Power Generation Schedule')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 图2：储热水平
            ax2 = axes[1]
            ax2.plot(time, op['E_TES'], 'b-', linewidth=2, marker='o')
            ax2.axhline(y=results['optimal_capacity']['E_TES_max'],
                        color='r', linestyle='--', label='Max Capacity')
            ax2.set_xlabel('Hour')
            ax2.set_ylabel('Energy (MWh)')
            ax2.set_title('Thermal Energy Storage Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 图3：弃能与电加热
            ax3 = axes[2]
            curtail = op['P_WC_WD'] + op['P_WC_PV']
            ax3.bar(time - 0.15, curtail, width=0.3, label='Curtailment', alpha=0.8, color='red')
            ax3.bar(time + 0.15, op['P_EH'], width=0.3, label='EH Absorption', alpha=0.8, color='green')
            ax3.set_xlabel('Hour')
            ax3.set_ylabel('Power (MW)')
            ax3.set_title('Curtailment and Electric Heater Absorption')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('capacity_planning_results.png', dpi=150)
            print("\n运行曲线已保存到 'capacity_planning_results.png'")
            plt.show()

        except ImportError:
            print("\n未安装matplotlib，跳过绘图")