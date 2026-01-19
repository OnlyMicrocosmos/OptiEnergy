"""
光伏-光热-风电-电加热器系统分布鲁棒优化模型
基于CVXPY实现，对应MATLAB second.txt / third.txt
【终极修复版】：引入变量缩放(Variable Scaling)解决数值病态问题
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum
import time


class ScenarioType(Enum):
    """天气场景类型"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"


@dataclass
class SystemParameters:
    """系统参数配置"""
    T: int = 24
    Delta_t: float = 1.0
    v_ci: float = 2.3
    v_r: float = 10.0
    v_co: float = 20.0
    P_r_single: float = 2.4
    eta_pv: float = 0.167
    S_pv_i: float = 2.3
    D_ref: float = 800.0
    T_ref: float = 20.0
    T_a: float = 25.0
    N_CT: float = 45.0
    eta_SF: float = 0.38
    gamma_TES: float = 0.038
    E_TES_min: float = 20.0
    E_init: float = 80.0
    eta_TES_cha: float = 0.98
    eta_TES_dis: float = 0.98
    Q_HTF_TES_max: float = 150.0
    Q_TES_HTF_max: float = 150.0
    eta_PB: float = 0.37
    P_PB_Min: float = 0.0
    P_PB_Max: float = 80.0
    Delta_P_Ru_PB: float = 50.0
    Delta_P_Rd_PB: float = 50.0
    T_min_On_PB: int = 1
    T_min_Off_PB: int = 1
    eta_EH: float = 0.95
    A_SF_min: float = 5000.0
    A_SF_max: float = 5e6
    E_TES_max_ub: float = 3000.0
    P_EH_Max_ub: float = 200.0
    P_wind_max: float = 500.0
    P_pv_max: float = 400.0


@dataclass
class CostParameters:
    """成本参数配置"""
    cost_SF_per_m2: float = 120.0
    cost_TES_per_kWh: float = 25.0
    cost_PB_per_kW: float = 880.0
    cost_PV_per_kW: float = 790.0
    cost_wind_per_kW: float = 1180.0
    cost_EH_per_kW: float = 40.0
    OM_CSP_per_kW: float = 24.0
    OM_PV_per_kW: float = 14.0
    OM_wind_per_kW: float = 50.0
    project_lifetime: int = 30
    discount_rate: float = 0.05
    mirror_maintenance: float = 7.82
    load_cut_penalty: float = 1000.0
    curtailment_penalty: float = 50.0


@dataclass
class DROResult:
    """分布鲁棒优化结果"""
    P_wind_capacity: float
    P_pv_capacity: float
    A_SF: float
    E_TES_max: float
    P_EH_Max: float
    P_CSP: np.ndarray
    P_WE_WD: np.ndarray
    P_WE_PV: np.ndarray
    P_WC_WD: np.ndarray
    P_WC_PV: np.ndarray
    P_EH: np.ndarray
    E_TES: np.ndarray
    load_cut: np.ndarray
    total_cost: float
    total_revenue: float
    total_profit: float
    worst_case_cost: float
    LCOE: float
    status: str
    solve_time: float


def get_default_data() -> Dict[str, np.ndarray]:
    """获取默认输入数据"""
    v_wind = np.array([6.47, 6.7, 6.57, 7.53, 7.33, 7.22, 7.4, 7.1, 6.82, 7.41,
                       6.07, 6.16, 6.84, 7.01, 6.77, 7.83, 7.32, 7.86, 7.53, 7.29,
                       6.59, 7.08, 6.61, 6.48])
    I_DNI = np.array([0, 0, 0, 0, 0, 50, 200, 350, 500, 750, 900, 1000,
                      1050, 950, 800, 700, 450, 200, 50, 0, 0, 0, 0, 0], dtype=float)
    price = np.array([45, 45, 45, 45, 45, 45, 45, 100, 100, 185, 185, 185,
                      185, 185, 100, 100, 185, 185, 185, 185, 185, 185, 45, 45], dtype=float)
    L_tt = np.array([23, 20, 32, 35, 40, 51, 56, 70, 64, 50, 45, 40,
                     49, 38, 42, 50, 71, 80, 84, 71, 60, 42, 24, 20], dtype=float)
    return {'v_wind': v_wind, 'I_DNI': I_DNI, 'price': price, 'L_tt': L_tt}


def calculate_unit_power(params: SystemParameters, v_wind: np.ndarray, I_DNI: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """计算单位风机功率和单位光伏功率"""
    T = len(v_wind)
    P_w_i = np.zeros(T)
    P_pv_unit = np.zeros(T)
    for t in range(T):
        if v_wind[t] < params.v_ci or v_wind[t] >= params.v_co:
            P_w_i[t] = 0
        elif params.v_ci <= v_wind[t] < params.v_r:
            P_w_i[t] = params.P_r_single * (v_wind[t] ** 3 - params.v_ci ** 3) / (params.v_r ** 3 - params.v_ci ** 3)
        else:
            P_w_i[t] = params.P_r_single
        T_pv = params.T_a + (params.N_CT - params.T_ref) * (I_DNI[t] / params.D_ref)
        P_pv_unit[t] = params.eta_pv * params.S_pv_i * I_DNI[t] / 1e6 * (1 - 0.005 * (T_pv - 25))
        P_pv_unit[t] = max(0, P_pv_unit[t])
    P_pv_single_STC = params.eta_pv * params.S_pv_i * params.D_ref / 1e6
    return P_w_i, P_pv_unit, P_pv_single_STC


def get_weather_scenario(I_DNI: np.ndarray) -> np.ndarray:
    """根据太阳辐射强度确定天气场景"""
    weather_scenario = np.zeros(len(I_DNI), dtype=int)
    for t, dni in enumerate(I_DNI):
        if dni >= 700:
            weather_scenario[t] = 1
        elif dni >= 200:
            weather_scenario[t] = 2
        else:
            weather_scenario[t] = 3
    return weather_scenario


def get_uncertainty_factor(scenario_type: ScenarioType) -> float:
    """根据天气场景获取不确定性因子"""
    factors = {ScenarioType.SUNNY: 0.5, ScenarioType.CLOUDY: 1.0, ScenarioType.RAINY: 1.5}
    return factors.get(scenario_type, 1.0)


def solve_dro_model(
        nominal_data: Dict[str, np.ndarray],
        rho: float,
        scenario_type: Optional[ScenarioType] = None,
        params: Optional[SystemParameters] = None,
        cost_params: Optional[CostParameters] = None,
        use_relaxed_binary: bool = True,
        verbose: bool = True
) -> DROResult:
    """求解分布鲁棒优化模型"""
    start_time = time.time()

    if params is None:
        params = SystemParameters()
    if cost_params is None:
        cost_params = CostParameters()

    T = params.T
    v_wind = nominal_data['v_wind']
    I_DNI = nominal_data['I_DNI']
    electricity_price = nominal_data['price']
    L_tt = nominal_data['L_tt']

    P_w_i, P_pv_unit, P_pv_single_STC = calculate_unit_power(params, v_wind, I_DNI)

    if scenario_type is not None:
        uncertainty_factor = get_uncertainty_factor(scenario_type)
        effective_rho = rho * uncertainty_factor
    else:
        effective_rho = rho

    weather_scenario = get_weather_scenario(I_DNI)
    time_varying_rho = np.zeros(T)
    for t in range(T):
        if weather_scenario[t] == 1:
            time_varying_rho[t] = effective_rho * 0.5
        elif weather_scenario[t] == 2:
            time_varying_rho[t] = effective_rho * 1.0
        else:
            time_varying_rho[t] = effective_rho * 1.5

    # ==================== 定义决策变量 (引入变量缩放) ====================
    # 【变量缩放技巧】：将大数值变量定义为小单位，例如1代表1万

    # 面积缩放因子：1.0 代表 10000 m2
    SF_SCALE = 10000.0
    A_SF_scaled = cp.Variable(nonneg=True, name="A_SF_scaled")
    # 物理面积（表达式）
    A_SF = A_SF_scaled * SF_SCALE

    P_wind_capacity = cp.Variable(nonneg=True, name="P_wind_capacity")
    P_pv_capacity = cp.Variable(nonneg=True, name="P_pv_capacity")

    E_TES_max = cp.Variable(nonneg=True, name="E_TES_max")
    P_EH_Max = cp.Variable(nonneg=True, name="P_EH_Max")

    P_PB = cp.Variable(T, nonneg=True, name="P_PB")
    P_CSP = cp.Variable(T, nonneg=True, name="P_CSP")
    P_WE_WD = cp.Variable(T, nonneg=True, name="P_WE_WD")
    P_WE_PV = cp.Variable(T, nonneg=True, name="P_WE_PV")
    P_WC_WD = cp.Variable(T, nonneg=True, name="P_WC_WD")
    P_WC_PV = cp.Variable(T, nonneg=True, name="P_WC_PV")
    P_EH = cp.Variable(T, nonneg=True, name="P_EH")
    load_cut = cp.Variable(T, nonneg=True, name="load_cut")

    E_TES = cp.Variable(T, nonneg=True, name="E_TES")
    Q_HTF_TES = cp.Variable(T, nonneg=True, name="Q_HTF_TES")
    Q_TES_HTF = cp.Variable(T, nonneg=True, name="Q_TES_HTF")
    Q_TES_cha = cp.Variable(T, nonneg=True, name="Q_TES_cha")
    Q_TES_dis = cp.Variable(T, nonneg=True, name="Q_TES_dis")
    Q_HTF_PB = cp.Variable(T, nonneg=True, name="Q_HTF_PB")
    Q_EH_HTF = cp.Variable(T, nonneg=True, name="Q_EH_HTF")
    Q_SF_in = cp.Variable(T, nonneg=True, name="Q_SF_in")
    Q_SF_HTF = cp.Variable(T, nonneg=True, name="Q_SF_HTF")
    Q_SF_cur = cp.Variable(T, nonneg=True, name="Q_SF_cur")
    Q_Loss = cp.Variable(T, nonneg=True, name="Q_Loss")

    P_wd_max = cp.Variable(T, nonneg=True, name="P_wd_max")
    P_pv_max = cp.Variable(T, nonneg=True, name="P_pv_max")

    if use_relaxed_binary:
        x_TES_cha = cp.Variable(T, nonneg=True, name="x_TES_cha")
        x_TES_dis = cp.Variable(T, nonneg=True, name="x_TES_dis")
    else:
        x_TES_cha = cp.Variable(T, boolean=True, name="x_TES_cha")
        x_TES_dis = cp.Variable(T, boolean=True, name="x_TES_dis")

    # 对偶变量
    lambda_wd = cp.Variable(T, nonneg=True, name="lambda_wd")
    lambda_pv = cp.Variable(T, nonneg=True, name="lambda_pv")

    # ==================== 约束条件 ====================
    constraints = []

    # 容量边界约束
    constraints.append(P_wind_capacity <= params.P_wind_max)
    constraints.append(P_pv_capacity <= params.P_pv_max)

    # 使用表达式 A_SF = A_SF_scaled * 10000 来进行约束
    constraints.append(A_SF >= params.A_SF_min)
    constraints.append(A_SF <= params.A_SF_max)

    constraints.append(E_TES_max <= params.E_TES_max_ub)
    constraints.append(P_EH_Max <= params.P_EH_Max_ub)

    # 理论最大功率计算
    for t in range(T):
        if P_w_i[t] > 1e-6:
            constraints.append(P_wd_max[t] == P_wind_capacity * P_w_i[t] / params.P_r_single)
        else:
            constraints.append(P_wd_max[t] == 0)
        if P_pv_single_STC > 1e-6 and P_pv_unit[t] > 1e-6:
            constraints.append(P_pv_max[t] == P_pv_capacity * P_pv_unit[t] / P_pv_single_STC)
        else:
            constraints.append(P_pv_max[t] == 0)

    # 鲁棒约束
    for t in range(T):
        rho_t = time_varying_rho[t]
        constraints.append(P_WE_WD[t] + P_WC_WD[t] <= P_wd_max[t] * (1 - rho_t))
        constraints.append(P_WE_PV[t] + P_WC_PV[t] <= P_pv_max[t] * (1 - rho_t))
        # 恢复正常的对偶约束
        constraints.append(lambda_wd[t] >= P_wd_max[t] * rho_t * cost_params.curtailment_penalty)
        constraints.append(lambda_pv[t] >= P_pv_max[t] * rho_t * cost_params.curtailment_penalty)

    # 热力循环回路能量平衡
    for t in range(T):
        constraints.append(Q_SF_HTF[t] + Q_TES_HTF[t] + Q_EH_HTF[t] == Q_HTF_TES[t] + Q_HTF_PB[t])

    # 负荷平衡约束
    for t in range(T):
        constraints.append(P_CSP[t] + P_WE_PV[t] + P_WE_WD[t] + load_cut[t] == L_tt[t])
        constraints.append(load_cut[t] <= L_tt[t])

    # 电加热
    for t in range(T):
        constraints.append(P_EH[t] <= P_WC_WD[t] + P_WC_PV[t])

    constraints.append(cp.sum(P_EH) >= 0.9 * cp.sum(P_WC_WD + P_WC_PV))
    constraints.append(cp.sum(load_cut) <= 0.1 * np.sum(L_tt))

    # 储热系统
    for t in range(T):
        constraints.append(Q_TES_cha[t] == params.eta_TES_cha * Q_HTF_TES[t])
        constraints.append(Q_TES_dis[t] == Q_TES_HTF[t] / params.eta_TES_dis)
        constraints.append(E_TES[t] >= params.E_TES_min)
        constraints.append(E_TES[t] <= E_TES_max)
        if t == 0:
            constraints.append(E_TES[t] == params.E_init)
        else:
            constraints.append(E_TES[t] == (1 - params.gamma_TES) * E_TES[t - 1] +
                               (Q_TES_cha[t] - Q_TES_dis[t]) * params.Delta_t)
        constraints.append(x_TES_cha[t] + x_TES_dis[t] <= 1)
        if use_relaxed_binary:
            constraints.append(x_TES_cha[t] <= 1)
            constraints.append(x_TES_dis[t] <= 1)
        constraints.append(Q_HTF_TES[t] <= params.Q_HTF_TES_max * x_TES_cha[t])
        constraints.append(Q_TES_HTF[t] <= params.Q_TES_HTF_max * x_TES_dis[t])

    # 发电机
    for t in range(T):
        constraints.append(P_PB[t] == params.eta_PB * Q_HTF_PB[t])
        constraints.append(P_PB[t] <= params.P_PB_Max)
        constraints.append(P_CSP[t] == P_PB[t])
        constraints.append(Q_EH_HTF[t] == params.eta_EH * P_EH[t])
        constraints.append(Q_EH_HTF[t] <= Q_HTF_TES[t] + Q_HTF_PB[t])
        constraints.append(P_EH[t] <= P_EH_Max)

    for t in range(1, T):
        constraints.append(P_PB[t] - P_PB[t - 1] <= params.Delta_P_Ru_PB)
        constraints.append(P_PB[t - 1] - P_PB[t] <= params.Delta_P_Rd_PB)

    # 太阳场
    for t in range(T):
        Q_SF_theoretical = params.eta_SF * I_DNI[t] * 1e-6
        constraints.append(Q_SF_in[t] <= Q_SF_theoretical * A_SF - Q_Loss[t])
        constraints.append(Q_SF_HTF[t] == Q_SF_in[t] - Q_SF_cur[t])

    daily_total_generation = cp.sum(P_PB + P_WE_WD + P_WE_PV)
    constraints.append(daily_total_generation <= np.sum(L_tt) * 2.0)

    # ==================== 目标函数 ====================
    # 目标函数缩放因子
    SCALE = 1e6

    # 投资成本 (A_SF 是表达式，会自动带入)
    cost_SF = cost_params.cost_SF_per_m2 * A_SF
    cost_TES = cost_params.cost_TES_per_kWh * 1000 * E_TES_max
    cost_PB = cost_params.cost_PB_per_kW * 1000 * params.P_PB_Max
    cost_PV = cost_params.cost_PV_per_kW * 1000 * P_pv_capacity
    cost_wind = cost_params.cost_wind_per_kW * 1000 * P_wind_capacity
    cost_EH = cost_params.cost_EH_per_kW * 1000 * P_EH_Max
    total_investment = cost_SF + cost_TES + cost_PB + cost_PV + cost_wind + cost_EH

    # 年运维
    OM_CSP = cost_params.OM_CSP_per_kW * 1000 * params.P_PB_Max
    OM_PV = cost_params.OM_PV_per_kW * 1000 * P_pv_capacity
    OM_wind = cost_params.OM_wind_per_kW * 1000 * P_wind_capacity
    mirror_maintenance = cost_params.mirror_maintenance * A_SF
    yearly_OM = OM_CSP + OM_PV + OM_wind + mirror_maintenance

    total_cost = total_investment + yearly_OM * cost_params.project_lifetime
    daily_revenue = cp.sum(cp.multiply(electricity_price, P_PB + P_WE_WD + P_WE_PV))
    load_cut_penalty_val = cost_params.load_cut_penalty * cp.sum(load_cut)
    curtailment_penalty_val = cost_params.curtailment_penalty * cp.sum(P_WC_WD + P_WC_PV - P_EH)
    worst_case_additional_cost = cp.sum(lambda_wd + lambda_pv) * 365 * cost_params.project_lifetime
    total_revenue = daily_revenue * 365 * cost_params.project_lifetime

    # 原始目标 (单位：美元)
    objective_raw = (total_revenue
                     - total_cost
                     - load_cut_penalty_val * 365 * cost_params.project_lifetime
                     - curtailment_penalty_val * 365 * cost_params.project_lifetime
                     - worst_case_additional_cost)

    # 【最大化缩放后的利润】
    problem = cp.Problem(cp.Maximize(objective_raw / SCALE), constraints)

    solve_success = False
    solvers_list = []

    # 优先 CLARABEL，其次 ECOS
    if 'CLARABEL' in cp.installed_solvers():
        solvers_list.append(cp.CLARABEL)
    if 'ECOS' in cp.installed_solvers():
        solvers_list.append(cp.ECOS)
    if 'SCS' in cp.installed_solvers():
        solvers_list.append(cp.SCS)

    for solver in solvers_list:
        try:
            if verbose:
                print(f"尝试求解器: {solver}")
            # 增加迭代次数
            problem.solve(solver=solver, verbose=False, max_iters=20000)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                solve_success = True
                if verbose:
                    print(f"求解器 {solver} 成功，状态: {problem.status}")
                break
        except Exception:
            continue

    if not solve_success:
        try:
            problem.solve(verbose=verbose)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                solve_success = True
        except Exception:
            pass

    solve_time = time.time() - start_time

    # ==================== 提取结果 ====================
    if problem.status in ['optimal', 'optimal_inaccurate']:
        def safe_val(var, default=0):
            if var is None:
                return default
            if hasattr(var, 'value'):
                if var.value is None:
                    return default if not hasattr(var, 'shape') else np.zeros(var.shape)
                return var.value
            return var

        P_PB_val = safe_val(P_PB, np.zeros(T))
        P_WE_WD_val = safe_val(P_WE_WD, np.zeros(T))
        P_WE_PV_val = safe_val(P_WE_PV, np.zeros(T))

        annual_gen = np.sum(P_PB_val + P_WE_WD_val + P_WE_PV_val) * params.Delta_t * 365

        total_investment_val = safe_val(total_investment)
        yearly_OM_val = safe_val(yearly_OM)

        numerator = total_investment_val
        denominator = 0
        for year in range(1, cost_params.project_lifetime + 1):
            numerator += yearly_OM_val / (1 + cost_params.discount_rate) ** year
            denominator += annual_gen / (1 + cost_params.discount_rate) ** year

        LCOE = numerator / max(denominator, 1e-6) / 1000.0

        actual_profit = problem.value * SCALE if problem.value is not None else 0

        result = DROResult(
            P_wind_capacity=safe_val(P_wind_capacity),
            P_pv_capacity=safe_val(P_pv_capacity),
            A_SF=safe_val(A_SF), # 这里获取的是 A_SF_scaled * 10000 的计算值
            E_TES_max=safe_val(E_TES_max),
            P_EH_Max=safe_val(P_EH_Max),
            P_CSP=safe_val(P_CSP, np.zeros(T)),
            P_WE_WD=P_WE_WD_val,
            P_WE_PV=P_WE_PV_val,
            P_WC_WD=safe_val(P_WC_WD, np.zeros(T)),
            P_WC_PV=safe_val(P_WC_PV, np.zeros(T)),
            P_EH=safe_val(P_EH, np.zeros(T)),
            E_TES=safe_val(E_TES, np.zeros(T)),
            load_cut=safe_val(load_cut, np.zeros(T)),
            total_cost=safe_val(total_cost),
            total_revenue=safe_val(total_revenue),
            total_profit=actual_profit,
            worst_case_cost=safe_val(worst_case_additional_cost),
            LCOE=LCOE,
            status=problem.status,
            solve_time=solve_time
        )
        return result
    else:
        return DROResult(0, 0, 0, 0, 0, np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T),
                         np.zeros(T), np.zeros(T), np.zeros(T), 0, 0, 0, 0, 0, problem.status, solve_time)


def run_sensitivity_analysis(nominal_data, rho_values=None, params=None, cost_params=None, verbose=True):
    if rho_values is None:
        rho_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'rho': rho_values, 'profit': [], 'wind_capacity': [], 'pv_capacity': [],
               'lcoe': [], 'load_cut': [], 'curtailment': []}
    for rho in rho_values:
        result = solve_dro_model(nominal_data, rho, params=params, cost_params=cost_params, verbose=False)
        if result.status in ['optimal', 'optimal_inaccurate']:
            results['profit'].append(result.total_profit / 1e6)
            results['wind_capacity'].append(result.P_wind_capacity)
            results['pv_capacity'].append(result.P_pv_capacity)
            results['lcoe'].append(result.LCOE * 1000)
            results['load_cut'].append(np.sum(result.load_cut))
            results['curtailment'].append(np.sum(result.P_WC_WD + result.P_WC_PV))
        else:
            for key in results.keys():
                if key != 'rho': results[key].append(np.nan)
    return results


def compare_scenarios(nominal_data, rho=0.1, params=None, cost_params=None, verbose=True):
    scenarios = [ScenarioType.SUNNY, ScenarioType.CLOUDY, ScenarioType.RAINY]
    results = {}
    for scenario in scenarios:
        result = solve_dro_model(nominal_data, rho, scenario_type=scenario, params=params,
                                 cost_params=cost_params, verbose=False)
        results[scenario.value] = result
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("分布鲁棒优化模型测试 (变量缩放修复版)")
    print("=" * 60)
    data = get_default_data()
    result = solve_dro_model(data, rho=0.1, verbose=True)
    if result.status in ['optimal', 'optimal_inaccurate']:
        print(f"\n测试通过! 解状态: {result.status}")
        print(f"  总利润: ${result.total_profit:,.2f}")
        print(f"  LCOE: ${result.LCOE:.4f}/kWh")
    else:
        print(f"\n测试失败，状态: {result.status}")
