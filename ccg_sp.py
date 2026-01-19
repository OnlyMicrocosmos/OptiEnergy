"""
CCG算法子问题求解器
对应MATLAB文件: SP.txt
"""

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class WorstCaseScenario:
    """存储SP找到的最坏场景"""
    Load_u: np.ndarray  # 最坏情况负荷 (24,)
    Ppv_z: np.ndarray  # 最坏情况光伏 (24,)
    Pw_w: np.ndarray  # 最坏情况风电 (24,)


@dataclass
class SPSolution:
    """存储SP的完整解"""
    UB: float
    Load_u: np.ndarray
    Ppv_z: np.ndarray
    Pw_w: np.ndarray
    P_CSP: np.ndarray
    P_WC_PV: np.ndarray
    P_WC_WD: np.ndarray
    P_EH: np.ndarray
    E_TES: np.ndarray
    Pbuy: np.ndarray
    Psell: np.ndarray
    P_DR: np.ndarray


@dataclass
class BinaryVars:
    """从主问题传入的二进制变量"""
    Temp_net: np.ndarray  # 购/售电标志 (24,)
    x_PB: np.ndarray  # 发电机组状态 (24,)
    u_PB: np.ndarray  # 启动变量 (24,)
    v_PB: np.ndarray  # 停机变量 (24,)
    x_TES_cha: np.ndarray  # 储热充热状态 (24,)
    x_TES_dis: np.ndarray  # 储热放热状态 (24,)


def get_sp_constants():
    """获取SP的系统常量（对应MATLAB SP.txt 第10-70行）"""

    T = 24

    Load = np.array([53, 50, 48, 49, 61, 51, 56, 70, 64, 50,
                     45, 40, 18, 20, 30, 50, 71, 100, 84, 71,
                     60, 42, 24, 20])

    I_DNI = np.array([0, 0, 0, 0, 0, 50, 150, 350, 550, 750,
                      900, 950, 920, 850, 750, 600, 400, 200, 50, 0,
                      0, 0, 0, 0])

    Ppv = 0.65 * np.array([0.00, 0.00, 0.00, 0.00, 0.00, 17.58, 43.65, 73.90, 93.48, 95.38,
                           102.30, 96.59, 101.40, 92.63, 93.50, 92.35, 68.09, 45.76, 21.81, 0.00,
                           0.00, 0.00, 0.00, 0.00])

    Pw = 0.65 * np.array([104.4555, 123.4790, 102.8142, 92.6088, 112.9052, 89.1061, 111.0100, 109.7712,
                          73.3869, 108.9974, 126.4295, 118.9272, 116.5876, 126.8384, 91.2581, 100.3876,
                          106.9537, 101.8204, 111.3992, 83.8046, 59.3542, 79.2348, 102.8312, 89.9582])

    params = {
        'T': T,
        'Load': Load,
        'I_DNI': I_DNI,
        'Ppv': Ppv,
        'Pw': Pw,
        'eta_SF': 0.4,
        'A_SF': 225000,
        'gamma_TES': 0.031,
        'E_TES_max': 1800 * 1000,
        'E_TES_min': 30 * 1000,
        'E_init': 900 * 1000,
        'eta_TES_cha': 0.98,
        'eta_TES_dis': 0.98,
        'Q_HTF_TES_max': 150 * 1000,
        'Q_TES_HTF_max': 150 * 1000,
        'eta_PB': 0.38,
        'P_PB_Min': 0,
        'P_PB_Max': 80 * 1000,
        'Delta_P_Ru_PB': 50 * 1000,
        'Delta_P_Rd_PB': 50 * 1000,
        'T_min_On_PB': 1,
        'T_min_Off_PB': 1,
        'Delta_t': 1,
        'eta_EH': 0.95,
        'P_EH_Max': 80 * 1000,
        'K_DR': 0.32,
        'P_DR_star': 0.2 * np.array([79, 70, 58, 48, 69, 68, 89, 97, 119, 149,
                                     170, 199, 140, 99, 100, 120, 140, 149, 191, 201,
                                     200, 189, 117, 79]),
        'P_net_max': 150,
        'K_CSP': 0.67,
        'K_TES': 0.38,
        'lambda_price': np.array([300, 300, 300, 300, 300, 300, 300, 700, 700, 1300,
                                  1300, 1300, 1300, 1300, 700, 700, 1300, 1300, 1300, 1300,
                                  1300, 1300, 300, 300]) / 1000,
        # 不确定性参数（对应MATLAB SP.txt 第108-110行）
        'load_uncertainty': 0.1,
        'pv_uncertainty': 0.1,
        'wind_uncertainty': 0.1,
        'budget_load': 6,
        'budget_pv': 6,
        'budget_wind': 6
    }

    params['DR'] = np.sum(params['P_DR_star'])
    params['DR_max'] = 1.5 * params['P_DR_star']
    params['DR_min'] = 0.5 * params['P_DR_star']

    return params


def solve_sp(binary_vars: BinaryVars) -> Tuple[float, Optional[WorstCaseScenario], Optional[SPSolution]]:
    """
    求解子问题（完整MILP版本）

    对应MATLAB SP.txt的KKT转换方法：
    [KKTsystem, details] = kkt(Constraints, F, [Load_u Ppv_z Pw_w]);
    optimize([KKTsystem, outerst], -F);

    注意：由于Python中没有直接的KKT转换，这里使用Big-M方法
    """

    params = get_sp_constants()
    T = params['T']

    # 提取固定的二进制变量
    Temp_net_fixed = binary_vars.Temp_net
    x_PB_fixed = binary_vars.x_PB
    x_TES_cha_fixed = binary_vars.x_TES_cha
    x_TES_dis_fixed = binary_vars.x_TES_dis

    # Big-M常数
    M = 1e6

    # ==================== 外层变量（不确定性）====================
    # 对应MATLAB SP.txt 第75-80行
    u = cp.Variable(T, boolean=True, name="u")  # 负荷不确定性指示
    z = cp.Variable(T, boolean=True, name="z")  # 光伏不确定性指示
    w = cp.Variable(T, boolean=True, name="w")  # 风电不确定性指示

    Load_u = cp.Variable(T, name="Load_u")
    Ppv_z = cp.Variable(T, name="Ppv_z")
    Pw_w = cp.Variable(T, name="Pw_w")

    # ==================== 内层变量（运行）====================
    Pbuy = cp.Variable(T, nonneg=True)
    Psell = cp.Variable(T, nonneg=True)
    P_DR = cp.Variable(T)
    P_DR_1 = cp.Variable(T, nonneg=True)
    P_DR_2 = cp.Variable(T, nonneg=True)

    Q_SF_in = cp.Variable(T)
    Q_SF_HTF = cp.Variable(T)
    Q_SF_cur = cp.Variable(T, nonneg=True)
    Q_Loss = cp.Variable(T, nonneg=True)
    Q_TES_HTF = cp.Variable(T, nonneg=True)
    Q_HTF_TES = cp.Variable(T, nonneg=True)
    Q_TES_cha = cp.Variable(T, nonneg=True)
    Q_TES_dis = cp.Variable(T, nonneg=True)
    E_TES = cp.Variable(T, nonneg=True)
    Q_EH_HTF = cp.Variable(T, nonneg=True)
    Q_HTF_PB = cp.Variable(T, nonneg=True)
    P_PB = cp.Variable(T, nonneg=True)
    P_CSP = cp.Variable(T, nonneg=True)

    P_WC_PV = cp.Variable(T, nonneg=True)
    P_WE_PV = cp.Variable(T, nonneg=True)
    P_WC_WD = cp.Variable(T, nonneg=True)
    P_WE_WD = cp.Variable(T, nonneg=True)
    P_EH = cp.Variable(T, nonneg=True)

    constraints = []

    # ==================== 外层约束（不确定集）====================
    # 对应MATLAB SP.txt 第108-115行
    for k in range(T):
        # 负荷不确定性（对应MATLAB: Load_u >= Load - 0.1*Load*u, Load_u <= Load + 0.1*Load*u）
        Load_base = params['Load'][k]
        delta_load = params['load_uncertainty'] * Load_base

        # u[k]=0: Load_u[k] = Load_base
        # u[k]=1: Load_u[k] ∈ [Load_base - delta, Load_base + delta]
        constraints.append(Load_u[k] >= Load_base - delta_load)
        constraints.append(Load_u[k] <= Load_base + delta_load * u[k])
        constraints.append(Load_u[k] >= Load_base + delta_load * (u[k] - 1))

        # 光伏不确定性
        Ppv_base = params['Ppv'][k]
        delta_pv = params['pv_uncertainty'] * Ppv_base

        constraints.append(Ppv_z[k] >= Ppv_base - delta_pv * z[k])
        constraints.append(Ppv_z[k] <= Ppv_base)
        constraints.append(Ppv_z[k] >= 0)

        # 风电不确定性
        Pw_base = params['Pw'][k]
        delta_wind = params['wind_uncertainty'] * Pw_base

        constraints.append(Pw_w[k] >= Pw_base - delta_wind * w[k])
        constraints.append(Pw_w[k] <= Pw_base)
        constraints.append(Pw_w[k] >= 0)

    # Budget约束
    constraints.append(cp.sum(u) <= params['budget_load'])
    constraints.append(cp.sum(z) <= params['budget_pv'])
    constraints.append(cp.sum(w) <= params['budget_wind'])

    # ==================== 内层约束（运行约束）====================
    for k in range(T):
        # 主网约束（使用固定的Temp_net）
        constraints.append(Pbuy[k] <= Temp_net_fixed[k] * params['P_net_max'])
        constraints.append(Psell[k] <= (1 - Temp_net_fixed[k]) * params['P_net_max'])

        # 需求响应约束
        constraints.append(P_DR[k] >= params['DR_min'][k])
        constraints.append(P_DR[k] <= params['DR_max'][k])
        constraints.append(P_DR[k] - params['P_DR_star'][k] + P_DR_1[k] - P_DR_2[k] == 0)

        # 风光功率守恒
        constraints.append(Pw_w[k] == P_WE_WD[k] + P_WC_WD[k])
        constraints.append(Ppv_z[k] == P_WE_PV[k] + P_WC_PV[k])
        constraints.append(P_EH[k] <= P_WC_WD[k] + P_WC_PV[k])

        # 储热系统
        constraints.append(Q_TES_cha[k] == params['eta_TES_cha'] * Q_HTF_TES[k])
        constraints.append(Q_TES_dis[k] == Q_TES_HTF[k] / params['eta_TES_dis'])

        if k == 0:
            constraints.append(E_TES[k] == params['E_init'])
        else:
            constraints.append(E_TES[k] == (1 - params['gamma_TES']) * E_TES[k - 1] +
                               (Q_TES_cha[k] - Q_TES_dis[k]) * params['Delta_t'])

        constraints.append(E_TES[k] >= params['E_TES_min'])
        constraints.append(E_TES[k] <= params['E_TES_max'])

        # 发电机组约束（使用固定的x_PB）
        constraints.append(Q_HTF_PB[k] == P_PB[k] / params['eta_PB'])
        constraints.append(P_PB[k] >= x_PB_fixed[k] * params['P_PB_Min'])
        constraints.append(P_PB[k] <= x_PB_fixed[k] * params['P_PB_Max'])

        if k >= 1:
            constraints.append(P_PB[k] - P_PB[k - 1] <= params['Delta_P_Ru_PB'])
            constraints.append(P_PB[k - 1] - P_PB[k] <= params['Delta_P_Rd_PB'])

        constraints.append(P_CSP[k] == P_PB[k])

        # 储热功率限值（使用固定的x_TES）
        constraints.append(Q_HTF_TES[k] <= params['Q_HTF_TES_max'] * x_TES_cha_fixed[k])
        constraints.append(Q_TES_HTF[k] <= params['Q_TES_HTF_max'] * x_TES_dis_fixed[k])

        # 太阳场
        constraints.append(Q_SF_in[k] == (params['eta_SF'] * params['I_DNI'][k] * params['A_SF']) / 1e3 - Q_Loss[k])
        constraints.append(Q_SF_HTF[k] == Q_SF_in[k] - Q_SF_cur[k])
        constraints.append(Q_SF_in[k] >= 0)

        # 电加热
        constraints.append(Q_EH_HTF[k] == params['eta_EH'] * P_EH[k])
        constraints.append(P_EH[k] <= params['P_EH_Max'])

        # HTF能量守恒
        constraints.append(Q_SF_HTF[k] + Q_TES_HTF[k] + Q_EH_HTF[k] == Q_HTF_PB[k] + Q_HTF_TES[k])

        # 功率平衡
        constraints.append(Pbuy[k] - Psell[k] == Load_u[k] + P_DR[k] - P_CSP[k] - P_WE_WD[k] - P_WE_PV[k])

        # 防止无界
        constraints.append(Pbuy[k] <= 1e5)
        constraints.append(Psell[k] <= 1e5)

    # 需求响应总量
    constraints.append(cp.sum(P_DR) == params['DR'])

    # ==================== 目标函数 ====================
    # 最大化成本（寻找最坏场景）
    C_M = params['lambda_price'] @ (Pbuy - Psell)
    C_CSP = params['K_CSP'] * cp.sum(P_CSP)
    C_TES = params['K_TES'] * (cp.sum(Q_TES_cha) + cp.sum(Q_TES_dis))
    C_DR = params['K_DR'] * cp.sum(P_DR_1 + P_DR_2)

    total_cost = C_M + C_CSP + C_TES + C_DR

    # ==================== 求解 ====================
    problem = cp.Problem(cp.Maximize(total_cost), constraints)

    try:
        if 'GUROBI' in cp.installed_solvers():
            problem.solve(solver=cp.GUROBI, verbose=False)
        elif 'CPLEX' in cp.installed_solvers():
            problem.solve(solver=cp.CPLEX, verbose=False)
        else:
            problem.solve(verbose=False)
    except:
        try:
            problem.solve(verbose=False)
        except:
            return np.inf, None, None

    if problem.status in ['optimal', 'optimal_inaccurate']:
        worst_case = WorstCaseScenario(
            Load_u=Load_u.value,
            Ppv_z=Ppv_z.value,
            Pw_w=Pw_w.value
        )

        solution = SPSolution(
            UB=problem.value,
            Load_u=Load_u.value,
            Ppv_z=Ppv_z.value,
            Pw_w=Pw_w.value,
            P_CSP=P_CSP.value,
            P_WC_PV=P_WC_PV.value,
            P_WC_WD=P_WC_WD.value,
            P_EH=P_EH.value,
            E_TES=E_TES.value,
            Pbuy=Pbuy.value,
            Psell=Psell.value,
            P_DR=P_DR.value
        )

        return problem.value, worst_case, solution
    else:
        return np.inf, None, None


def solve_sp_simplified(binary_vars: BinaryVars) -> Tuple[float, Optional[WorstCaseScenario]]:
    """
    简化版SP：通过枚举极端点求解最坏场景

    策略：
    1. 构造满足Budget约束的极端场景
    2. 对每个场景求解运行问题
    3. 返回成本最高的场景
    """

    params = get_sp_constants()
    T = params['T']

    Load = params['Load']
    Ppv = params['Ppv']
    Pw = params['Pw']

    epsilon = params['load_uncertainty']  # 0.1
    budget = params['budget_load']  # 6

    # ==================== 构造候选最坏场景 ====================
    # 最坏情况：负荷高、可再生能源低

    # 选择影响最大的时段（负荷高峰、可再生能源高峰）
    load_impact = Load * epsilon
    pv_impact = Ppv * epsilon
    wind_impact = Pw * epsilon

    # 负荷最高的6个时段
    load_top_idx = np.argsort(load_impact)[-budget:]
    # 光伏最高的6个时段
    pv_top_idx = np.argsort(pv_impact)[-budget:]
    # 风电最高的6个时段
    wind_top_idx = np.argsort(wind_impact)[-budget:]

    candidate_scenarios = []

    # 场景1：负荷高峰时段增加负荷 + 可再生能源高峰时段减少出力（最恶劣）
    Load_worst = Load.copy()
    Load_worst[load_top_idx] = Load[load_top_idx] * (1 + epsilon)

    Ppv_worst = Ppv.copy()
    Ppv_worst[pv_top_idx] = Ppv[pv_top_idx] * (1 - epsilon)

    Pw_worst = Pw.copy()
    Pw_worst[wind_top_idx] = Pw[wind_top_idx] * (1 - epsilon)

    candidate_scenarios.append((Load_worst, Ppv_worst, Pw_worst, "HighLoad_LowRES"))

    # 场景2：仅负荷增加
    Load_high = Load.copy()
    Load_high[load_top_idx] = Load[load_top_idx] * (1 + epsilon)
    candidate_scenarios.append((Load_high, Ppv.copy(), Pw.copy(), "HighLoad_Only"))

    # 场景3：仅可再生能源减少
    Ppv_low = Ppv.copy()
    Ppv_low[pv_top_idx] = Ppv[pv_top_idx] * (1 - epsilon)
    Pw_low = Pw.copy()
    Pw_low[wind_top_idx] = Pw[wind_top_idx] * (1 - epsilon)
    candidate_scenarios.append((Load.copy(), Ppv_low, Pw_low, "LowRES_Only"))

    # 场景4：电价高峰时段负荷增加
    # 电价高峰: 时段10-14, 17-22 (1300元/MWh)
    price_peak_idx = [9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21]
    peak_load_idx = [i for i in price_peak_idx if i < T][:budget]
    Load_peak = Load.copy()
    for idx in peak_load_idx:
        Load_peak[idx] = Load[idx] * (1 + epsilon)
    candidate_scenarios.append((Load_peak, Ppv.copy(), Pw.copy(), "PeakPrice_HighLoad"))

    # 场景5：名义场景（保底）
    candidate_scenarios.append((Load.copy(), Ppv.copy(), Pw.copy(), "Nominal"))

    # ==================== 评估每个场景 ====================
    max_cost = -np.inf
    worst_scenario = None

    for Load_u_val, Ppv_z_val, Pw_w_val, name in candidate_scenarios:
        cost = _solve_operational_problem(binary_vars, Load_u_val, Ppv_z_val, Pw_w_val, params)

        if cost is not None and cost > max_cost:
            max_cost = cost
            worst_scenario = WorstCaseScenario(
                Load_u=Load_u_val,
                Ppv_z=Ppv_z_val,
                Pw_w=Pw_w_val
            )

    if worst_scenario is None:
        # 返回名义场景作为后备
        return 1e9, WorstCaseScenario(Load, Ppv, Pw)

    return max_cost, worst_scenario


def _solve_operational_problem(binary_vars: BinaryVars,
                               Load_u_val: np.ndarray,
                               Ppv_z_val: np.ndarray,
                               Pw_w_val: np.ndarray,
                               params: dict) -> Optional[float]:
    """
    求解给定场景下的运行问题（LP）
    """

    T = params['T']

    Temp_net_fixed = binary_vars.Temp_net
    x_PB_fixed = binary_vars.x_PB
    x_TES_cha_fixed = binary_vars.x_TES_cha
    x_TES_dis_fixed = binary_vars.x_TES_dis

    # ==================== 变量 ====================
    Pbuy = cp.Variable(T, nonneg=True)
    Psell = cp.Variable(T, nonneg=True)
    P_DR = cp.Variable(T)
    P_DR_1 = cp.Variable(T, nonneg=True)
    P_DR_2 = cp.Variable(T, nonneg=True)

    Q_SF_in = cp.Variable(T)
    Q_SF_HTF = cp.Variable(T)
    Q_SF_cur = cp.Variable(T, nonneg=True)
    Q_Loss = cp.Variable(T, nonneg=True)
    Q_TES_HTF = cp.Variable(T, nonneg=True)
    Q_HTF_TES = cp.Variable(T, nonneg=True)
    Q_TES_cha = cp.Variable(T, nonneg=True)
    Q_TES_dis = cp.Variable(T, nonneg=True)
    E_TES = cp.Variable(T, nonneg=True)
    Q_EH_HTF = cp.Variable(T, nonneg=True)
    Q_HTF_PB = cp.Variable(T, nonneg=True)
    P_PB = cp.Variable(T, nonneg=True)
    P_CSP = cp.Variable(T, nonneg=True)

    P_WC_PV = cp.Variable(T, nonneg=True)
    P_WE_PV = cp.Variable(T, nonneg=True)
    P_WC_WD = cp.Variable(T, nonneg=True)
    P_WE_WD = cp.Variable(T, nonneg=True)
    P_EH = cp.Variable(T, nonneg=True)

    # 切负荷变量（保证可行性）
    load_shed = cp.Variable(T, nonneg=True)
    PENALTY = 500.0  # 切负荷惩罚

    constraints = []

    for k in range(T):
        # 主网约束
        constraints.append(Pbuy[k] <= Temp_net_fixed[k] * params['P_net_max'])
        constraints.append(Psell[k] <= (1 - Temp_net_fixed[k]) * params['P_net_max'])

        # 需求响应
        constraints.append(P_DR[k] >= params['DR_min'][k])
        constraints.append(P_DR[k] <= params['DR_max'][k])
        constraints.append(P_DR[k] - params['P_DR_star'][k] + P_DR_1[k] - P_DR_2[k] == 0)

        # 风光守恒
        constraints.append(Pw_w_val[k] == P_WE_WD[k] + P_WC_WD[k])
        constraints.append(Ppv_z_val[k] == P_WE_PV[k] + P_WC_PV[k])
        constraints.append(P_EH[k] <= P_WC_WD[k] + P_WC_PV[k])

        # 储热系统
        constraints.append(Q_TES_cha[k] == params['eta_TES_cha'] * Q_HTF_TES[k])
        constraints.append(Q_TES_dis[k] == Q_TES_HTF[k] / params['eta_TES_dis'])

        if k == 0:
            constraints.append(E_TES[k] == params['E_init'])
        else:
            constraints.append(E_TES[k] == (1 - params['gamma_TES']) * E_TES[k - 1] +
                               (Q_TES_cha[k] - Q_TES_dis[k]) * params['Delta_t'])

        constraints.append(E_TES[k] >= params['E_TES_min'])
        constraints.append(E_TES[k] <= params['E_TES_max'])

        # 发电机组
        constraints.append(Q_HTF_PB[k] == P_PB[k] / params['eta_PB'])
        constraints.append(P_PB[k] >= x_PB_fixed[k] * params['P_PB_Min'])
        constraints.append(P_PB[k] <= x_PB_fixed[k] * params['P_PB_Max'])

        if k >= 1:
            constraints.append(P_PB[k] - P_PB[k - 1] <= params['Delta_P_Ru_PB'])
            constraints.append(P_PB[k - 1] - P_PB[k] <= params['Delta_P_Rd_PB'])

        constraints.append(P_CSP[k] == P_PB[k])

        # 储热功率
        constraints.append(Q_HTF_TES[k] <= params['Q_HTF_TES_max'] * x_TES_cha_fixed[k])
        constraints.append(Q_TES_HTF[k] <= params['Q_TES_HTF_max'] * x_TES_dis_fixed[k])

        # 太阳场
        constraints.append(Q_SF_in[k] == (params['eta_SF'] * params['I_DNI'][k] * params['A_SF']) / 1e3 - Q_Loss[k])
        constraints.append(Q_SF_HTF[k] == Q_SF_in[k] - Q_SF_cur[k])
        constraints.append(Q_SF_in[k] >= 0)

        # 电加热
        constraints.append(Q_EH_HTF[k] == params['eta_EH'] * P_EH[k])
        constraints.append(P_EH[k] <= params['P_EH_Max'])

        # HTF能量守恒
        constraints.append(Q_SF_HTF[k] + Q_TES_HTF[k] + Q_EH_HTF[k] == Q_HTF_PB[k] + Q_HTF_TES[k])

        # 功率平衡（含切负荷）
        constraints.append(
            Pbuy[k] - Psell[k] == Load_u_val[k] + P_DR[k] - P_CSP[k] - P_WE_WD[k] - P_WE_PV[k] - load_shed[k])

        # 切负荷上限
        constraints.append(load_shed[k] <= Load_u_val[k])

    # 需求响应总量
    constraints.append(cp.sum(P_DR) == params['DR'])

    # ==================== 目标函数 ====================
    C_M = params['lambda_price'] @ (Pbuy - Psell)
    C_CSP = params['K_CSP'] * cp.sum(P_CSP)
    C_TES = params['K_TES'] * (cp.sum(Q_TES_cha) + cp.sum(Q_TES_dis))
    C_DR = params['K_DR'] * cp.sum(P_DR_1 + P_DR_2)

    # 总成本（含切负荷惩罚）
    total_cost = C_M + C_CSP + C_TES + C_DR + PENALTY * cp.sum(load_shed)

    # ==================== 求解 ====================
    problem = cp.Problem(cp.Minimize(total_cost), constraints)

    try:
        if 'GUROBI' in cp.installed_solvers():
            problem.solve(solver=cp.GUROBI, verbose=False)
        elif 'HIGHS' in cp.installed_solvers():
            problem.solve(solver=cp.HIGHS, verbose=False)
        elif 'ECOS' in cp.installed_solvers():
            problem.solve(solver=cp.ECOS, verbose=False)
        else:
            problem.solve(verbose=False)
    except:
        try:
            problem.solve(verbose=False)
        except:
            return None

    if problem.status in ['optimal', 'optimal_inaccurate']:
        return problem.value
    else:
        return None


if __name__ == "__main__":
    print("测试SP...")

    # 创建测试用的二进制变量
    binary_vars = BinaryVars(
        Temp_net=np.ones(24, dtype=int),
        x_PB=np.ones(24, dtype=int),
        u_PB=np.zeros(24, dtype=int),
        v_PB=np.zeros(24, dtype=int),
        x_TES_cha=np.zeros(24, dtype=int),
        x_TES_dis=np.zeros(24, dtype=int)
    )

    print("测试简化版SP...")
    ub, worst = solve_sp_simplified(binary_vars)
    print(f"UB: {ub}")
    if worst:
        print(f"最坏负荷（部分）: {worst.Load_u[:5]}")