"""
CCG算法主问题求解器
对应MATLAB文件: MP.txt 和 MP_start.m
"""

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


@dataclass
class ScheduleSolution:
    """存储主问题的解"""
    Temp_net: np.ndarray  # 购/售电标志 (24,)
    x_PB: np.ndarray  # 发电机组运行状态 (24,)
    u_PB: np.ndarray  # 启动变量 (24,)
    v_PB: np.ndarray  # 停机变量 (24,)
    x_TES_cha: np.ndarray  # 储热充热状态 (24,)
    x_TES_dis: np.ndarray  # 储热放热状态 (24,)
    Pbuy: np.ndarray  # 从电网购电 (24,)
    Psell: np.ndarray  # 向电网售电 (24,)
    P_CSP: np.ndarray  # 光热电站输出 (24,)
    P_PB: np.ndarray  # 发电机组输出 (24,)
    E_TES: np.ndarray  # 储热能量 (24,)
    P_WC_PV: np.ndarray  # 弃光量 (24,)
    P_WC_WD: np.ndarray  # 弃风量 (24,)
    P_WE_PV: np.ndarray  # 光伏上网功率 (24,)
    P_WE_WD: np.ndarray  # 风电上网功率 (24,)
    P_EH: np.ndarray  # 电加热器功率 (24,)
    P_DR: np.ndarray  # 需求响应功率 (24,)
    Q_HTF_TES: np.ndarray  # HTF到储热的热功率 (24,)
    Q_TES_HTF: np.ndarray  # 储热到HTF的热功率 (24,)
    Q_SF_HTF: np.ndarray  # 太阳场到HTF的热功率 (24,)
    Q_EH_HTF: np.ndarray  # 电加热到HTF的热功率 (24,)
    Q_HTF_PB: np.ndarray  # HTF到发电机组的热功率 (24,)


@dataclass
class ScenarioData:
    """存储最坏场景数据"""
    Load_u: np.ndarray  # 最坏情况负荷 (24,)
    Ppv_z: np.ndarray  # 最坏情况光伏 (24,)
    Pw_w: np.ndarray  # 最坏情况风电 (24,)


def get_system_constants():
    """获取系统常量（对应MATLAB MP.txt 第10-60行）"""

    T = 24  # 时间段数

    # 基础数据
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

    # 系统参数
    params = {
        'T': T,
        'Load': Load,
        'I_DNI': I_DNI,
        'Ppv': Ppv,
        'Pw': Pw,
        'eta_SF': 0.4,
        'A_SF': 225000,
        'gamma_TES': 0.031,
        'E_TES_max': 1800 * 1000,  # kWh
        'E_TES_min': 30 * 1000,  # kWh
        'E_init': 900 * 1000,  # kWh
        'eta_TES_cha': 0.98,
        'eta_TES_dis': 0.98,
        'Q_HTF_TES_max': 150 * 1000,  # kW
        'Q_TES_HTF_max': 150 * 1000,  # kW
        'eta_PB': 0.38,
        'P_PB_Min': 0,
        'P_PB_Max': 80 * 1000,  # kW
        'Delta_P_Ru_PB': 50 * 1000,  # kW/h
        'Delta_P_Rd_PB': 50 * 1000,  # kW/h
        'T_min_On_PB': 1,
        'T_min_Off_PB': 1,
        'Delta_t': 1,
        'eta_EH': 0.95,
        'P_EH_Max': 80 * 1000,  # kW
        'K_DR': 0.32,
        'P_DR_star': 0.2 * np.array([79, 70, 58, 48, 69, 68, 89, 97, 119, 149,
                                     170, 199, 140, 99, 100, 120, 140, 149, 191, 201,
                                     200, 189, 117, 79]),
        'P_net_max': 150,  # kW
        'K_CSP': 0.67,
        'K_TES': 0.38,
        'lambda_price': np.array([300, 300, 300, 300, 300, 300, 300, 700, 700, 1300,
                                  1300, 1300, 1300, 1300, 700, 700, 1300, 1300, 1300, 1300,
                                  1300, 1300, 300, 300]) / 1000
    }

    params['DR'] = np.sum(params['P_DR_star'])
    params['DR_max'] = 1.5 * params['P_DR_star']
    params['DR_min'] = 0.5 * params['P_DR_star']

    return params


def solve_mp_start() -> Tuple[float, Optional[ScheduleSolution]]:
    """
    求解初始主问题（使用名义场景）
    对应MATLAB: MP_start.m
    """
    params = get_system_constants()

    # 使用名义场景
    Load_u = params['Load']
    Ppv_z = params['Ppv']
    Pw_w = params['Pw']

    return _solve_mp_core(params, Load_u, Ppv_z, Pw_w, UB=np.inf)


def solve_mp(UB: float, Load_u: np.ndarray, Ppv_z: np.ndarray,
             Pw_w: np.ndarray) -> Tuple[float, Optional[ScheduleSolution]]:
    """
    求解主问题（使用最坏场景）
    对应MATLAB: function [LB, Temp_net] = MP(UB, Load_u, Ppv_z, Pw_w)
    """
    params = get_system_constants()
    return _solve_mp_core(params, Load_u, Ppv_z, Pw_w, UB)


def _solve_mp_core(params: dict, Load_u: np.ndarray, Ppv_z: np.ndarray,
                   Pw_w: np.ndarray, UB: float) -> Tuple[float, Optional[ScheduleSolution]]:
    """
    主问题核心求解函数
    对应MATLAB MP.txt 的完整逻辑
    """

    T = params['T']

    # ==================== 变量定义（对应MATLAB MP.txt 第3-50行） ====================
    # 连续变量
    Pbuy = cp.Variable(T, nonneg=True, name="Pbuy")
    Psell = cp.Variable(T, nonneg=True, name="Psell")
    P_DR = cp.Variable(T, name="P_DR")
    P_DR_1 = cp.Variable(T, nonneg=True, name="P_DR_1")
    P_DR_2 = cp.Variable(T, nonneg=True, name="P_DR_2")

    # 光热电站变量
    Q_SF_in = cp.Variable(T, name="Q_SF_in")
    Q_SF_HTF = cp.Variable(T, name="Q_SF_HTF")
    Q_SF_cur = cp.Variable(T, nonneg=True, name="Q_SF_cur")
    Q_Loss = cp.Variable(T, nonneg=True, name="Q_Loss")
    Q_TES_HTF = cp.Variable(T, nonneg=True, name="Q_TES_HTF")
    Q_HTF_TES = cp.Variable(T, nonneg=True, name="Q_HTF_TES")
    Q_TES_cha = cp.Variable(T, nonneg=True, name="Q_TES_cha")
    Q_TES_dis = cp.Variable(T, nonneg=True, name="Q_TES_dis")
    E_TES = cp.Variable(T, nonneg=True, name="E_TES")
    Q_EH_HTF = cp.Variable(T, nonneg=True, name="Q_EH_HTF")
    Q_HTF_PB = cp.Variable(T, nonneg=True, name="Q_HTF_PB")
    P_PB = cp.Variable(T, nonneg=True, name="P_PB")
    P_CSP = cp.Variable(T, nonneg=True, name="P_CSP")

    # 风电光伏变量
    P_WC_PV = cp.Variable(T, nonneg=True, name="P_WC_PV")
    P_WE_PV = cp.Variable(T, nonneg=True, name="P_WE_PV")
    P_WC_WD = cp.Variable(T, nonneg=True, name="P_WC_WD")
    P_WE_WD = cp.Variable(T, nonneg=True, name="P_WE_WD")
    P_EH = cp.Variable(T, nonneg=True, name="P_EH")

    # 二进制变量
    Temp_net = cp.Variable(T, boolean=True, name="Temp_net")
    x_PB = cp.Variable(T, boolean=True, name="x_PB")
    u_PB = cp.Variable(T, boolean=True, name="u_PB")
    v_PB = cp.Variable(T, boolean=True, name="v_PB")
    x_TES_cha = cp.Variable(T, boolean=True, name="x_TES_cha")
    x_TES_dis = cp.Variable(T, boolean=True, name="x_TES_dis")

    # ==================== 约束条件（对应MATLAB MP.txt 第65-150行） ====================
    constraints = []

    for k in range(T):
        # 主网约束（对应MATLAB第68-69行）
        constraints.append(Pbuy[k] <= Temp_net[k] * params['P_net_max'])
        constraints.append(Psell[k] <= (1 - Temp_net[k]) * params['P_net_max'])

        # 需求响应约束（对应MATLAB第72-75行）
        constraints.append(P_DR[k] >= params['DR_min'][k])
        constraints.append(P_DR[k] <= params['DR_max'][k])
        constraints.append(P_DR[k] - params['P_DR_star'][k] + P_DR_1[k] - P_DR_2[k] == 0)

        # 风电光伏功率守恒（对应MATLAB第78-83行）
        constraints.append(Pw_w[k] == P_WE_WD[k] + P_WC_WD[k])
        constraints.append(Ppv_z[k] == P_WE_PV[k] + P_WC_PV[k])
        constraints.append(P_EH[k] <= P_WC_WD[k] + P_WC_PV[k])

        # 储热系统关系（对应MATLAB第88-98行）
        constraints.append(Q_TES_cha[k] == params['eta_TES_cha'] * Q_HTF_TES[k])
        constraints.append(Q_TES_dis[k] == Q_TES_HTF[k] / params['eta_TES_dis'])

        if k == 0:
            constraints.append(E_TES[k] == params['E_init'])
        else:
            constraints.append(E_TES[k] == (1 - params['gamma_TES']) * E_TES[k - 1] +
                               (Q_TES_cha[k] - Q_TES_dis[k]) * params['Delta_t'])

        constraints.append(E_TES[k] >= params['E_TES_min'])
        constraints.append(E_TES[k] <= params['E_TES_max'])

        # 发电机组约束（对应MATLAB第101-113行）
        constraints.append(Q_HTF_PB[k] == P_PB[k] / params['eta_PB'])
        constraints.append(P_PB[k] >= x_PB[k] * params['P_PB_Min'])
        constraints.append(P_PB[k] <= x_PB[k] * params['P_PB_Max'])

        if k >= 1:
            # 爬坡约束
            constraints.append(P_PB[k] - P_PB[k - 1] + x_PB[k - 1] * (params['P_PB_Min'] - params['Delta_P_Ru_PB']) +
                               x_PB[k] * (params['P_PB_Max'] - params['P_PB_Min']) <= params['P_PB_Max'])
            constraints.append(P_PB[k - 1] - P_PB[k] + x_PB[k] * (params['P_PB_Min'] - params['Delta_P_Rd_PB']) +
                               x_PB[k - 1] * (params['P_PB_Max'] - params['P_PB_Min']) <= params['P_PB_Max'])

        constraints.append(P_CSP[k] == P_PB[k])

        # 储热功率限值（对应MATLAB第120-121行）
        constraints.append(Q_HTF_TES[k] <= params['Q_HTF_TES_max'] * x_TES_cha[k])
        constraints.append(Q_TES_HTF[k] <= params['Q_TES_HTF_max'] * x_TES_dis[k])

        # 太阳场约束（对应MATLAB第124-127行）
        constraints.append(Q_SF_in[k] == (params['eta_SF'] * params['I_DNI'][k] * params['A_SF']) / 1e3 - Q_Loss[k])
        constraints.append(Q_SF_HTF[k] == Q_SF_in[k] - Q_SF_cur[k])
        constraints.append(Q_SF_in[k] >= 0)

        # 电加热约束（对应MATLAB第130-131行）
        constraints.append(Q_EH_HTF[k] == params['eta_EH'] * P_EH[k])
        constraints.append(P_EH[k] <= params['P_EH_Max'])

        # HTF能量守恒（对应MATLAB第134行）
        constraints.append(Q_SF_HTF[k] + Q_TES_HTF[k] + Q_EH_HTF[k] == Q_HTF_PB[k] + Q_HTF_TES[k])

        # 储热充放互斥（对应MATLAB第137行）
        constraints.append(x_TES_cha[k] + x_TES_dis[k] <= 1)

        # 功率平衡（对应MATLAB第140行）
        constraints.append(Pbuy[k] - Psell[k] == Load_u[k] + P_DR[k] - P_CSP[k] - P_WE_WD[k] - P_WE_PV[k])

    # 机组最短开停机时间约束（对应MATLAB第144-150行）
    T_min_On = params['T_min_On_PB']
    T_min_Off = params['T_min_Off_PB']

    for k in range(T_min_On + 1, T):
        constraints.append(cp.sum(x_PB[k - T_min_On:k]) >= v_PB[k] * T_min_On)

    for k in range(T_min_Off + 1, T):
        constraints.append(cp.sum(1 - x_PB[k - T_min_Off:k]) >= u_PB[k] * T_min_Off)

    # 初始状态（对应MATLAB第153行）
    constraints.append(x_PB[0] == 1)

    # 启停逻辑约束（对应MATLAB第156-159行）
    for k in range(1, T):
        constraints.append(x_PB[k] - x_PB[k - 1] == u_PB[k] - v_PB[k])
        constraints.append(u_PB[k] + v_PB[k] <= 1)

    # 需求响应总量约束（对应MATLAB第161行）
    constraints.append(cp.sum(P_DR) == params['DR'])

    # ==================== 目标函数（对应MATLAB MP.txt 第165-178行） ====================
    # 电网交互成本
    C_M = params['lambda_price'] @ (Pbuy - Psell)

    # 光热电站发电成本
    C_CSP = params['K_CSP'] * cp.sum(P_CSP)

    # 储热系统成本
    C_TES = params['K_TES'] * (cp.sum(Q_TES_cha) + cp.sum(Q_TES_dis))

    # 需求响应成本
    C_DR = params['K_DR'] * cp.sum(P_DR_1 + P_DR_2)

    # 总成本
    obj = C_M + C_CSP + C_TES + C_DR

    # 添加上界约束（对应MATLAB第181行）
    if np.isfinite(UB):
        constraints.append(obj <= UB)

    # ==================== 求解 ====================
    problem = cp.Problem(cp.Minimize(obj), constraints)

    # 尝试求解器
    solvers_to_try = []
    if 'GUROBI' in cp.installed_solvers():
        solvers_to_try.append(cp.GUROBI)
    if 'CPLEX' in cp.installed_solvers():
        solvers_to_try.append(cp.CPLEX)
    if 'HIGHS' in cp.installed_solvers():
        solvers_to_try.append(cp.HIGHS)
    if 'CBC' in cp.installed_solvers():
        solvers_to_try.append(cp.CBC)
    if 'GLPK_MI' in cp.installed_solvers():
        solvers_to_try.append(cp.GLPK_MI)
    if 'SCIP' in cp.installed_solvers():
        solvers_to_try.append(cp.SCIP)

    solved = False
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                solved = True
                break
        except:
            continue

    if not solved:
        try:
            problem.solve(verbose=False)
        except Exception as e:
            print(f"MP求解失败: {e}")
            return np.inf, None

    # ==================== 提取解 ====================
    if problem.status in ['optimal', 'optimal_inaccurate']:
        def val(var):
            return var.value if var.value is not None else np.zeros(T)

        solution = ScheduleSolution(
            Temp_net=np.round(val(Temp_net)).astype(int),
            x_PB=np.round(val(x_PB)).astype(int),
            u_PB=np.round(val(u_PB)).astype(int),
            v_PB=np.round(val(v_PB)).astype(int),
            x_TES_cha=np.round(val(x_TES_cha)).astype(int),
            x_TES_dis=np.round(val(x_TES_dis)).astype(int),
            Pbuy=val(Pbuy),
            Psell=val(Psell),
            P_CSP=val(P_CSP),
            P_PB=val(P_PB),
            E_TES=val(E_TES),
            P_WC_PV=val(P_WC_PV),
            P_WC_WD=val(P_WC_WD),
            P_WE_PV=val(P_WE_PV),
            P_WE_WD=val(P_WE_WD),
            P_EH=val(P_EH),
            P_DR=val(P_DR),
            Q_HTF_TES=val(Q_HTF_TES),
            Q_TES_HTF=val(Q_TES_HTF),
            Q_SF_HTF=val(Q_SF_HTF),
            Q_EH_HTF=val(Q_EH_HTF),
            Q_HTF_PB=val(Q_HTF_PB)
        )
        return problem.value, solution
    else:
        print(f"MP优化失败，状态: {problem.status}")
        return np.inf, None


if __name__ == "__main__":
    print("测试MP_start...")
    lb, sol = solve_mp_start()
    print(f"初始LB: {lb}")
    if sol is not None:
        print(f"Temp_net: {sol.Temp_net}")