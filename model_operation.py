"""
能源系统优化模型 - 基于CVXPY实现
包含：光热电站(CSP)、光伏(PV)、风电(Wind)、电加热(EH)、储热系统(TES)
"""

import cvxpy as cp
import numpy as np
from typing import Dict, Any, Optional


def solve_operation_model(
        load_data: np.ndarray,
        res_data: Dict[str, np.ndarray],
        params: Dict[str, Any],
        flags: Dict[str, bool]
) -> Dict[str, Any]:
    """
    能源系统日前优化调度模型

    Parameters:
    -----------
    load_data : np.ndarray
        24小时负荷数据 (MW)，形状为 (24,)

    res_data : dict
        可再生能源出力数据，包含:
        - 'pv_power': 光伏出力数组 (24,) (MW)
        - 'wind_power': 风电出力数组 (24,) (MW)

    params : dict
        系统参数，包含:
        - 'rho': 电价数组 (24,) (元/MWh)
        - 'I_DNI': 太阳直射辐射数组 (24,) (W/m²)
        - 'eta_SF': 光热转化效率
        - 'A_SF': 太阳场集热面积 (m²)
        - 'gamma_TES': 储热损失系数
        - 'E_TES_max': 最大储热能量 (MWh)
        - 'E_TES_min': 最小储热能量 (MWh)
        - 'E_init': 初始储热能量 (MWh)
        - 'eta_TES_cha': 储热充热效率
        - 'eta_TES_dis': 储热放热效率
        - 'Q_HTF_TES_max': 最大吸热功率 (MW)
        - 'Q_TES_HTF_max': 最大放热功率 (MW)
        - 'eta_PB': 热电转换效率
        - 'P_PB_min': 发电机组最小出力 (MW)
        - 'P_PB_max': 发电机组最大出力 (MW)
        - 'Delta_P_Ru_PB': 上爬坡速率 (MW/h)
        - 'Delta_P_Rd_PB': 下爬坡速率 (MW/h)
        - 'eta_EH': 电加热效率
        - 'P_EH_max': 电加热最大功率 (MW)
        - 'load_penalty': 欠负荷惩罚系数 (元/MWh)
        - 'curtail_penalty': 弃能惩罚系数 (元/MWh)

    flags : dict
        功能开关:
        - 'use_pv': 是否使用光伏
        - 'use_wind': 是否使用风电
        - 'use_eh': 是否使用电加热

    Returns:
    --------
    dict : 优化结果，包含各变量的最优值和统计信息
    """

    # ==================== 1. 参数提取 ====================
    T = 24  # 时间段数
    Delta_t = 1  # 时间间隔（小时）

    # 负荷数据
    L_tt = np.array(load_data).flatten()

    # 功能开关
    use_pv = flags.get('use_pv', True)
    use_wind = flags.get('use_wind', True)
    use_eh = flags.get('use_eh', True)

    # 可再生能源数据
    raw_pv = np.array(res_data.get('pv_power', np.zeros(T))).flatten()
    raw_wd = np.array(res_data.get('wind_power', np.zeros(T))).flatten()

    # 如果禁用了组件，将对应的资源输入强制设为 0，防止约束冲突
    P_pv = raw_pv if use_pv else np.zeros(T)
    P_wd = raw_wd if use_wind else np.zeros(T)

    # 系统参数 - 使用默认值
    rho = np.array(params.get('rho', np.ones(T) * 500)).flatten()
    I_DNI = np.array(params.get('I_DNI', np.zeros(T))).flatten()

    # CSP系统参数
    eta_SF = params.get('eta_SF', 0.4)
    A_SF = params.get('A_SF', 225000)
    gamma_TES = params.get('gamma_TES', 0.031)
    E_TES_max = params.get('E_TES_max', 1800)
    E_TES_min = params.get('E_TES_min', 30)
    E_init = params.get('E_init', 900)
    eta_TES_cha = params.get('eta_TES_cha', 0.98)
    eta_TES_dis = params.get('eta_TES_dis', 0.98)
    Q_HTF_TES_max = params.get('Q_HTF_TES_max', 150)
    Q_TES_HTF_max = params.get('Q_TES_HTF_max', 150)
    eta_PB = params.get('eta_PB', 0.38)
    P_PB_min = params.get('P_PB_min', 0)
    P_PB_max = params.get('P_PB_max', 80)
    Delta_P_Ru_PB = params.get('Delta_P_Ru_PB', 50)
    Delta_P_Rd_PB = params.get('Delta_P_Rd_PB', 50)

    # 电加热参数
    eta_EH = params.get('eta_EH', 0.95)
    P_EH_max = params.get('P_EH_max', 80)

    # 惩罚系数
    load_penalty = params.get('load_penalty', 5000)
    curtail_penalty = params.get('curtail_penalty', 300)



    # ==================== 2. 定义决策变量 ====================

    # --- CSP系统变量 ---
    Q_SF_in = cp.Variable(T, nonneg=True)  # 太阳场净热功率 (MW)
    Q_SF_HTF = cp.Variable(T, nonneg=True)  # 太阳场向HTF提供的热功率 (MW)
    Q_SF_cur = cp.Variable(T, nonneg=True)  # 太阳场弃热功率 (MW)
    Q_Loss = cp.Variable(T, nonneg=True)  # 热损失功率 (MW)

    # --- 储热系统变量 ---
    Q_TES_HTF = cp.Variable(T, nonneg=True)  # 储热向HTF放热功率 (MW)
    Q_HTF_TES = cp.Variable(T, nonneg=True)  # HTF向储热充热功率 (MW)
    Q_TES_cha = cp.Variable(T, nonneg=True)  # 储热充热功率(考虑效率后) (MW)
    Q_TES_dis = cp.Variable(T, nonneg=True)  # 储热放热功率(考虑效率前) (MW)
    E_TES = cp.Variable(T, nonneg=True)  # 储热能量 (MWh)

    # --- 发电机组变量 ---
    Q_HTF_PB = cp.Variable(T, nonneg=True)  # HTF传递给发电机组的热功率 (MW)
    P_PB = cp.Variable(T, nonneg=True)  # 发电机组电功率输出 (MW)
    P_CSP = cp.Variable(T, nonneg=True)  # 光热电站净发电功率 (MW)

    # --- 二值变量 (发电机组状态) ---
    x_PB = cp.Variable(T, boolean=True)  # 发电机组运行状态
    x_TES_cha = cp.Variable(T, boolean=True)  # 储热充热状态
    x_TES_dis = cp.Variable(T, boolean=True)  # 储热放热状态

    # --- 电加热器变量 ---
    Q_EH_HTF = cp.Variable(T, nonneg=True)  # 电加热器传递的热功率 (MW)
    P_EH = cp.Variable(T, nonneg=True)  # 电加热器电功率 (MW)

    # --- 风电变量 ---
    P_WE_WD = cp.Variable(T, nonneg=True)  # 风电上网功率 (MW)
    P_WC_WD = cp.Variable(T, nonneg=True)  # 弃风量 (MW)

    # --- 光伏变量 ---
    P_WE_PV = cp.Variable(T, nonneg=True)  # 光伏上网功率 (MW)
    P_WC_PV = cp.Variable(T, nonneg=True)  # 弃光量 (MW)

    # --- 负荷变量 ---
    load_shed = cp.Variable(T, nonneg=True)  # 欠负荷量 (MW)

    # ==================== 3. 构造约束 ====================
    constraints = []

    # 大M值用于线性化
    M = 1e6

    for t in range(T):
        # --- 3.1 HTF能量平衡约束 ---
        # Q_SF_HTF + Q_TES_HTF + Q_EH_HTF == Q_HTF_TES + Q_HTF_PB
        constraints.append(
            Q_SF_HTF[t] + Q_TES_HTF[t] + Q_EH_HTF[t] == Q_HTF_TES[t] + Q_HTF_PB[t]
        )

        # --- 3.2 太阳场约束 ---
        # 太阳场净热功率 = 太阳辐射转换热量 - 热损失
        Q_SF_theoretical = (eta_SF * I_DNI[t] * A_SF) / 1e6  # 转换为MW
        constraints.append(Q_SF_in[t] == Q_SF_theoretical - Q_Loss[t])
        constraints.append(Q_SF_HTF[t] == Q_SF_in[t] - Q_SF_cur[t])

        # --- 3.3 储热系统约束 ---
        # 充放热功率与效率关系
        constraints.append(Q_TES_cha[t] == eta_TES_cha * Q_HTF_TES[t])
        constraints.append(Q_TES_dis[t] == Q_TES_HTF[t] / eta_TES_dis)

        # 储热能量动态方程
        if t == 0:
            constraints.append(E_TES[t] == E_init)
        else:
            constraints.append(
                E_TES[t] == (1 - gamma_TES) * E_TES[t - 1] +
                (Q_TES_cha[t] - Q_TES_dis[t]) * Delta_t
            )

        # 储热能量上下限
        constraints.append(E_TES[t] >= E_TES_min)
        constraints.append(E_TES[t] <= E_TES_max)

        # 充放热功率限制（与二值变量关联）
        constraints.append(Q_HTF_TES[t] <= Q_HTF_TES_max * x_TES_cha[t])
        constraints.append(Q_TES_HTF[t] <= Q_TES_HTF_max * x_TES_dis[t])

        # 充放热互斥
        constraints.append(x_TES_cha[t] + x_TES_dis[t] <= 1)

        # --- 3.4 发电机组约束 ---
        # 热电转换
        constraints.append(Q_HTF_PB[t] == P_PB[t] / eta_PB)

        # 出力上下限（与运行状态关联）
        constraints.append(P_PB[t] >= P_PB_min * x_PB[t])
        constraints.append(P_PB[t] <= P_PB_max * x_PB[t])

        # 爬坡约束
        if t >= 1:
            constraints.append(P_PB[t] - P_PB[t - 1] <= Delta_P_Ru_PB)
            constraints.append(P_PB[t - 1] - P_PB[t] <= Delta_P_Rd_PB)

        # CSP净发电等于发电机组出力
        constraints.append(P_CSP[t] == P_PB[t])

        # --- 3.5 电加热约束 ---
        # 电热转换
        constraints.append(Q_EH_HTF[t] == eta_EH * P_EH[t])

        # 电加热功率上限
        constraints.append(P_EH[t] <= P_EH_max)

        # 电加热只能使用弃风弃光
        constraints.append(P_EH[t] <= P_WC_WD[t] + P_WC_PV[t])

        # --- 3.6 风电约束 ---
        # 风电功率守恒
        constraints.append(P_WE_WD[t] + P_WC_WD[t] == P_wd[t])

        # --- 3.7 光伏约束 ---
        # 光伏功率守恒
        constraints.append(P_WE_PV[t] + P_WC_PV[t] == P_pv[t])

        # --- 3.8 负荷平衡约束 ---
        constraints.append(
            P_CSP[t] + P_WE_WD[t] + P_WE_PV[t] + load_shed[t] == L_tt[t]
        )

        # 欠负荷上限
        constraints.append(load_shed[t] <= L_tt[t])

    # --- 3.9 初始状态约束 ---
    constraints.append(x_PB[0] == 1)  # 初始开机

    # ==================== 4. 根据flags添加额外约束 ====================

    # 如果不使用风电
    if not use_wind:
        for t in range(T):
            constraints.append(P_WE_WD[t] == 0)
            constraints.append(P_WC_WD[t] == 0)

    # 如果不使用光伏
    if not use_pv:
        for t in range(T):
            constraints.append(P_WE_PV[t] == 0)
            constraints.append(P_WC_PV[t] == 0)

    # 如果不使用电加热
    if not use_eh:
        for t in range(T):
            constraints.append(P_EH[t] == 0)
            constraints.append(Q_EH_HTF[t] == 0)

    # ==================== 5. 目标函数 ====================
    # 最大化收益 = 售电收入 - 欠负荷惩罚 - 弃能惩罚

    # 售电收入
    revenue = cp.sum(cp.multiply(rho, P_PB + P_WE_WD + P_WE_PV))

    # 欠负荷惩罚
    load_shed_cost = load_penalty * cp.sum(load_shed)

    # 弃能惩罚（扣除电加热消纳的部分）
    curtail_cost = curtail_penalty * cp.sum(P_WC_WD + P_WC_PV - P_EH)

    # 总目标：最大化收益（等价于最小化负收益）
    objective = cp.Maximize(revenue - load_shed_cost - curtail_cost)

    # ==================== 6. 求解模型 ====================
    problem = cp.Problem(objective, constraints)

    # 尝试不同的求解器
    solvers_to_try = []

    # 检查可用的求解器
    if 'CBC' in cp.installed_solvers():
        solvers_to_try.append(cp.CBC)
    if 'SCIP' in cp.installed_solvers():
        solvers_to_try.append(cp.SCIP)
    if 'GLPK_MI' in cp.installed_solvers():
        solvers_to_try.append(cp.GLPK_MI)
    if 'HIGHS' in cp.installed_solvers():
        solvers_to_try.append(cp.HIGHS)
    if 'ECOS_BB' in cp.installed_solvers():
        solvers_to_try.append(cp.ECOS_BB)

    # 如果没有找到MIP求解器，添加默认选项
    if not solvers_to_try:
        solvers_to_try = [None]  # 使用CVXPY默认求解器

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

    # ==================== 7. 结果提取 ====================
    if solve_success:
        results = {
            'status': 'optimal',
            'message': solve_info,
            'objective_value': problem.value,

            # CSP系统
            'P_CSP': P_CSP.value,
            'P_PB': P_PB.value,
            'x_PB': x_PB.value,
            'Q_SF_in': Q_SF_in.value,
            'Q_SF_HTF': Q_SF_HTF.value,
            'Q_SF_cur': Q_SF_cur.value,
            'Q_Loss': Q_Loss.value,
            'Q_HTF_PB': Q_HTF_PB.value,

            # 储热系统
            'E_TES': E_TES.value,
            'Q_TES_cha': Q_TES_cha.value,
            'Q_TES_dis': Q_TES_dis.value,
            'Q_HTF_TES': Q_HTF_TES.value,
            'Q_TES_HTF': Q_TES_HTF.value,
            'x_TES_cha': x_TES_cha.value,
            'x_TES_dis': x_TES_dis.value,

            # 电加热
            'P_EH': P_EH.value,
            'Q_EH_HTF': Q_EH_HTF.value,

            # 风电
            'P_WE_WD': P_WE_WD.value,
            'P_WC_WD': P_WC_WD.value,

            # 光伏
            'P_WE_PV': P_WE_PV.value,
            'P_WC_PV': P_WC_PV.value,

            # 负荷
            'load_shed': load_shed.value,
            'load_demand': L_tt,

            # 统计信息
            'statistics': {
                'total_csp_generation': np.sum(P_CSP.value) if P_CSP.value is not None else 0,
                'total_wind_grid': np.sum(P_WE_WD.value) if P_WE_WD.value is not None else 0,
                'total_pv_grid': np.sum(P_WE_PV.value) if P_WE_PV.value is not None else 0,
                'total_wind_curtail': np.sum(P_WC_WD.value) if P_WC_WD.value is not None else 0,
                'total_pv_curtail': np.sum(P_WC_PV.value) if P_WC_PV.value is not None else 0,
                'total_eh_consumption': np.sum(P_EH.value) if P_EH.value is not None else 0,
                'total_load_shed': np.sum(load_shed.value) if load_shed.value is not None else 0,
                'total_revenue': problem.value,
                'wind_utilization_rate': (
                    (np.sum(P_WE_WD.value) / np.sum(P_wd) * 100)
                    if np.sum(P_wd) > 0 and P_WE_WD.value is not None else 0
                ),
                'pv_utilization_rate': (
                    (np.sum(P_WE_PV.value) / np.sum(P_pv) * 100)
                    if np.sum(P_pv) > 0 and P_WE_PV.value is not None else 0
                ),
            }
        }
    else:
        results = {
            'status': 'failed',
            'message': f"求解失败: {solve_info}",
            'objective_value': None,
        }

    return results


def get_default_params() -> Dict[str, Any]:
    """
    获取默认参数配置

    Returns:
    --------
    dict : 默认参数字典
    """
    return {
        # 电价 (元/MWh)
        'rho': np.array([300, 300, 300, 300, 300, 300, 300, 700, 700, 1300,
                         1300, 1300, 1300, 1300, 700, 700, 1300, 1300, 1300,
                         1300, 1300, 1300, 300, 300]),

        # 太阳直射辐射 (W/m²)
        'I_DNI': np.array([0, 0, 0, 0, 0, 0, 0, 0.0725, 4.5052, 59.4297,
                           173.3273, 779.3091, 447.2624, 551.1209, 704.1094,
                           366.4278, 531.1645, 236.8347, 89.0775, 57.2798,
                           5.2521, 0, 0, 0]),

        # CSP系统参数
        'eta_SF': 0.4,  # 光热转化效率
        'A_SF': 225000,  # 太阳场面积 (m²)
        'gamma_TES': 0.031,  # 储热损失系数
        'E_TES_max': 1800,  # 最大储热能量 (MWh)
        'E_TES_min': 30,  # 最小储热能量 (MWh)
        'E_init': 900,  # 初始储热能量 (MWh)
        'eta_TES_cha': 0.98,  # 储热充热效率
        'eta_TES_dis': 0.98,  # 储热放热效率
        'Q_HTF_TES_max': 150,  # 最大吸热功率 (MW)
        'Q_TES_HTF_max': 150,  # 最大放热功率 (MW)
        'eta_PB': 0.38,  # 热电转换效率
        'P_PB_min': 0,  # 发电机组最小出力 (MW)
        'P_PB_max': 80,  # 发电机组最大出力 (MW)
        'Delta_P_Ru_PB': 50,  # 上爬坡速率 (MW/h)
        'Delta_P_Rd_PB': 50,  # 下爬坡速率 (MW/h)

        # 电加热参数
        'eta_EH': 0.95,  # 电加热效率
        'P_EH_max': 80,  # 电加热最大功率 (MW)

        # 惩罚系数
        'load_penalty': 5000,  # 欠负荷惩罚 (元/MWh)
        'curtail_penalty': 300,  # 弃能惩罚 (元/MWh)
    }


def print_results(results: Dict[str, Any]) -> None:
    """
    打印优化结果

    Parameters:
    -----------
    results : dict
        优化结果字典
    """
    if results['status'] != 'optimal':
        print(f"优化失败: {results['message']}")
        return

    print("\n" + "=" * 60)
    print("                    优化结果汇总")
    print("=" * 60)

    stats = results['statistics']

    print(f"\n{'目标函数值 (总收益):':<30} {results['objective_value']:>15.2f} 元")

    print("\n--- 发电统计 ---")
    print(f"{'CSP总发电量:':<30} {stats['total_csp_generation']:>15.2f} MWh")
    print(f"{'风电上网总量:':<30} {stats['total_wind_grid']:>15.2f} MWh")
    print(f"{'光伏上网总量:':<30} {stats['total_pv_grid']:>15.2f} MWh")

    print("\n--- 弃能统计 ---")
    print(f"{'弃风总量:':<30} {stats['total_wind_curtail']:>15.2f} MWh")
    print(f"{'弃光总量:':<30} {stats['total_pv_curtail']:>15.2f} MWh")
    print(f"{'电加热消纳量:':<30} {stats['total_eh_consumption']:>15.2f} MWh")
    actual_curtail = stats['total_wind_curtail'] + stats['total_pv_curtail'] - stats['total_eh_consumption']
    print(f"{'实际弃能量:':<30} {actual_curtail:>15.2f} MWh")

    print("\n--- 利用率 ---")
    print(f"{'风电利用率:':<30} {stats['wind_utilization_rate']:>14.2f} %")
    print(f"{'光伏利用率:':<30} {stats['pv_utilization_rate']:>14.2f} %")

    print("\n--- 负荷统计 ---")
    print(f"{'欠负荷总量:':<30} {stats['total_load_shed']:>15.2f} MWh")

    print("\n" + "=" * 60)

    # 打印各时段详细数据
    print("\n--- 各时段详细数据 ---")
    print(f"{'时段':<6} {'CSP出力':<10} {'风电上网':<10} {'光伏上网':<10} "
          f"{'电加热':<10} {'储热能量':<12} {'欠负荷':<10}")
    print("-" * 70)

    for t in range(24):
        csp = results['P_CSP'][t] if results['P_CSP'] is not None else 0
        wind = results['P_WE_WD'][t] if results['P_WE_WD'] is not None else 0
        pv = results['P_WE_PV'][t] if results['P_WE_PV'] is not None else 0
        eh = results['P_EH'][t] if results['P_EH'] is not None else 0
        tes = results['E_TES'][t] if results['E_TES'] is not None else 0
        shed = results['load_shed'][t] if results['load_shed'] is not None else 0

        print(f"{t + 1:<6} {csp:<10.2f} {wind:<10.2f} {pv:<10.2f} "
              f"{eh:<10.2f} {tes:<12.2f} {shed:<10.2f}")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试用例：复现all.txt的场景

    # 负荷数据
    L_st = np.array([53, 50, 48, 49, 61, 51, 56, 70, 64, 50, 45, 40,
                     18, 20, 30, 50, 71, 100, 84, 71, 60, 42, 24, 20])
    load_data = 0.7 * L_st

    # 风电出力数据
    P_wd_1 = np.array([12.0689, 8.5662, 12.4262, 17.4022, 9.8139, 12.0633,
                       5.9847, 13.8371, 10.4123, 4.9822, 4.2232, 6.2094,
                       11.3067, 33.0847, 21.6094, 12.0326, 16.0073, 9.3553,
                       13.8792, 4.7155, 13.7928, 9.0076, 13.5399, 9.8148])
    wind_power = 0.45 * 5 * P_wd_1

    # 光伏出力数据
    P_pv_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0.001184159, 0.220677134,
                       2.041627643, 7.324570355, 32.28073353, 16.66810552,
                       21.19112019, 22.45754812, 15.75532432, 18.56005547,
                       6.96306104, 1.598958202, 0.785685838, 0.069879846,
                       0, 0, 0])
    pv_power = 0.45 * 5 * P_pv_1

    # 可再生能源数据
    res_data = {
        'pv_power': pv_power,
        'wind_power': wind_power
    }

    # 获取默认参数
    params = get_default_params()

    # 场景1: 全部组件启用 (对应 all.txt)
    print("\n" + "=" * 60)
    print("场景1: 风光储热 + 电加热 (全组件)")
    print("=" * 60)

    flags_all = {
        'use_pv': True,
        'use_wind': True,
        'use_eh': True
    }

    results_all = solve_operation_model(load_data, res_data, params, flags_all)
    print_results(results_all)

    # 场景2: 仅光伏 + 电加热 (对应 PV.txt)
    print("\n" + "=" * 60)
    print("场景2: 仅光伏 + 电加热")
    print("=" * 60)

    flags_pv = {
        'use_pv': True,
        'use_wind': False,
        'use_eh': True
    }

    results_pv = solve_operation_model(load_data, res_data, params, flags_pv)
    print_results(results_pv)

    # 场景3: 仅风电 + 电加热 (对应 WINDd.txt)
    print("\n" + "=" * 60)
    print("场景3: 仅风电 + 电加热")
    print("=" * 60)

    flags_wind = {
        'use_pv': False,
        'use_wind': True,
        'use_eh': True
    }

    results_wind = solve_operation_model(load_data, res_data, params, flags_wind)
    print_results(results_wind)

    # 场景4: 风光储热，无电加热 (对应 noeh 文件)
    print("\n" + "=" * 60)
    print("场景4: 风光储热，无电加热")
    print("=" * 60)

    flags_noeh = {
        'use_pv': True,
        'use_wind': True,
        'use_eh': False
    }

    results_noeh = solve_operation_model(load_data, res_data, params, flags_noeh)
    print_results(results_noeh)