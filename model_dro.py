"""
光伏-光热-风电-电加热器系统分布鲁棒优化模型
基于CVXPY实现，对应MATLAB second.txt / third.txt
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
    # 时间参数
    T: int = 24  # 时间段数
    Delta_t: float = 1.0  # 时间间隔（小时）

    # 风力发电参数
    v_ci: float = 2.3  # 切入风速 (m/s)
    v_r: float = 10.0  # 额定风速 (m/s)
    v_co: float = 20.0  # 切出风速 (m/s)
    P_r_single: float = 2.4  # 单台风机额定功率 (MW)

    # 光伏发电参数
    eta_pv: float = 0.167  # 光伏效率
    S_pv_i: float = 2.3  # 单个光伏组件面积 (m²)
    D_ref: float = 800.0  # 参考辐射强度 (W/m²)
    T_ref: float = 20.0  # 参考温度 (℃)
    T_a: float = 25.0  # 标准测试条件下的测试温度 (℃)
    N_CT: float = 45.0  # 额定运行温度 (℃)

    # 光热系统参数
    eta_SF: float = 0.38  # 光热转化效率
    gamma_TES: float = 0.038  # 储热系统热损失系数
    E_TES_min: float = 20.0  # 最小储热能量 (MWh)
    E_init: float = 80.0  # 初始储热能量 (MWh)
    eta_TES_cha: float = 0.98  # 储热充热效率
    eta_TES_dis: float = 0.98  # 储热放热效率
    Q_HTF_TES_max: float = 150.0  # 最大吸热功率 (MW)
    Q_TES_HTF_max: float = 150.0  # 最大放热功率 (MW)
    eta_PB: float = 0.37  # 热电转换效率
    P_PB_Min: float = 0.0  # 发电机组最小出力 (MW)
    P_PB_Max: float = 80.0  # 发电机组最大出力 (MW)
    Delta_P_Ru_PB: float = 50.0  # 上调速率 (MW/h)
    Delta_P_Rd_PB: float = 50.0  # 下调速率 (MW/h)
    T_min_On_PB: int = 1  # 最短开机时间 (小时)
    T_min_Off_PB: int = 1  # 最短停机时间 (小时)
    eta_EH: float = 0.95  # 电加热设备电转热效率

    # 容量边界（对应MATLAB second.txt 第245-249行）
    A_SF_min: float = 5000.0  # 镜场面积下限 (m²)
    A_SF_max: float = 5e6  # 镜场面积上限 (m²)
    E_TES_max_ub: float = 3000.0  # 储热容量上限 (MWh)
    P_EH_Max_ub: float = 200.0  # 电加热容量上限 (MW)
    P_wind_max: float = 500.0  # 风电容量上限 (MW)
    P_pv_max: float = 400.0  # 光伏容量上限 (MW)


@dataclass
class CostParameters:
    """成本参数配置"""
    # 投资成本
    cost_SF_per_m2: float = 120.0  # 镜场成本 (美元/m²)
    cost_TES_per_kWh: float = 25.0  # 储热成本 (美元/kWh)
    cost_PB_per_kW: float = 880.0  # 发电机组成本 (美元/kW)
    cost_PV_per_kW: float = 790.0  # 光伏投资成本 (美元/kW)
    cost_wind_per_kW: float = 1180.0  # 风电投资成本 (美元/kW)
    cost_EH_per_kW: float = 40.0  # 电加热器成本 (美元/kW)

    # 年运维成本
    OM_CSP_per_kW: float = 24.0  # 光热系统年运维成本 (美元/kW/年)
    OM_PV_per_kW: float = 14.0  # 光伏系统年运维成本 (美元/kW/年)
    OM_wind_per_kW: float = 50.0  # 风电系统年运维成本 (美元/kW/年)

    # 其他参数
    project_lifetime: int = 30  # 项目寿命 (年)
    discount_rate: float = 0.05  # 折现率
    mirror_maintenance: float = 7.82  # 镜场年维护成本 (美元/m²/年)

    # 惩罚成本
    load_cut_penalty: float = 1000.0  # 切负荷惩罚 (美元/MWh)
    curtailment_penalty: float = 50.0  # 弃风弃光惩罚 (美元/MWh)


@dataclass
class DROResult:
    """分布鲁棒优化结果"""
    # 容量优化结果
    P_wind_capacity: float  # 风电容量 (MW)
    P_pv_capacity: float  # 光伏容量 (MW)
    A_SF: float  # 镜场面积 (m²)
    E_TES_max: float  # 储热容量 (MWh)
    P_EH_Max: float  # 电加热容量 (MW)

    # 运行调度结果
    P_CSP: np.ndarray  # 光热电站输出 (T,)
    P_WE_WD: np.ndarray  # 风电上网功率 (T,)
    P_WE_PV: np.ndarray  # 光伏上网功率 (T,)
    P_WC_WD: np.ndarray  # 弃风量 (T,)
    P_WC_PV: np.ndarray  # 弃光量 (T,)
    P_EH: np.ndarray  # 电加热器功率 (T,)
    E_TES: np.ndarray  # 储热能量 (T,)
    load_cut: np.ndarray  # 切负荷 (T,)

    # 经济指标
    total_cost: float  # 总成本 (美元)
    total_revenue: float  # 总收益 (美元)
    total_profit: float  # 总利润 (美元)
    worst_case_cost: float  # 最坏情况成本 (美元)
    LCOE: float  # 平准化度电成本 (美元/kWh)

    # 求解状态
    status: str  # 求解状态
    solve_time: float  # 求解时间 (秒)


def get_default_data() -> Dict[str, np.ndarray]:
    """获取默认输入数据"""
    # 风速数据 (m/s)
    v_wind = np.array([6.47, 6.7, 6.57, 7.53, 7.33, 7.22, 7.4, 7.1, 6.82, 7.41,
                       6.07, 6.16, 6.84, 7.01, 6.77, 7.83, 7.32, 7.86, 7.53, 7.29,
                       6.59, 7.08, 6.61, 6.48])

    # 太阳辐射强度数据 (W/m²)
    I_DNI = np.array([0, 0, 0, 0, 0, 50, 200, 350, 500, 750, 900, 1000,
                      1050, 950, 800, 700, 450, 200, 50, 0, 0, 0, 0, 0], dtype=float)

    # 电价数据 (美元/MWh)
    price = np.array([45, 45, 45, 45, 45, 45, 45, 100, 100, 185, 185, 185,
                      185, 185, 100, 100, 185, 185, 185, 185, 185, 185, 45, 45], dtype=float)

    # 负荷需求数据 (MW)
    L_tt = np.array([23, 20, 32, 35, 40, 51, 56, 70, 64, 50, 45, 40,
                     49, 38, 42, 50, 71, 80, 84, 71, 60, 42, 24, 20], dtype=float)

    return {
        'v_wind': v_wind,
        'I_DNI': I_DNI,
        'price': price,
        'L_tt': L_tt
    }


def calculate_unit_power(params: SystemParameters,
                         v_wind: np.ndarray,
                         I_DNI: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算单位风机功率和单位光伏功率

    Args:
        params: 系统参数
        v_wind: 风速数据 (T,)
        I_DNI: 太阳辐射强度数据 (T,)

    Returns:
        P_w_i: 单台风机功率 (T,)
        P_pv_unit: 单位光伏功率 (T,)
        P_pv_single_STC: 单个光伏组件在标准条件下的功率 (MW)
    """
    T = len(v_wind)
    P_w_i = np.zeros(T)
    P_pv_unit = np.zeros(T)

    for t in range(T):
        # 风力发电计算 - 单台风机
        if v_wind[t] < params.v_ci or v_wind[t] >= params.v_co:
            P_w_i[t] = 0
        elif params.v_ci <= v_wind[t] < params.v_r:
            P_w_i[t] = params.P_r_single * (v_wind[t] ** 3 - params.v_ci ** 3) / (params.v_r ** 3 - params.v_ci ** 3)
        else:
            P_w_i[t] = params.P_r_single

        # 光伏发电计算
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
            weather_scenario[t] = 1  # 晴天
        elif dni >= 200:
            weather_scenario[t] = 2  # 多云
        else:
            weather_scenario[t] = 3  # 阴天/夜间
    return weather_scenario


def get_uncertainty_factor(scenario_type: ScenarioType) -> float:
    """根据天气场景获取不确定性因子"""
    factors = {
        ScenarioType.SUNNY: 0.5,
        ScenarioType.CLOUDY: 1.0,
        ScenarioType.RAINY: 1.5
    }
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
    """
    求解分布鲁棒优化模型

    基于Box Uncertainty的鲁棒优化，考虑风光出力的不确定性。
    通过对偶变换将min-max问题转化为单层优化问题。

    Args:
        nominal_data: 包含以下键的字典
            - 'v_wind': 风速预测数据 (T,)
            - 'I_DNI': 太阳辐射强度预测数据 (T,)
            - 'price': 电价数据 (T,)
            - 'L_tt': 负荷需求数据 (T,)
        rho: 不确定度半径 (0到1之间)
        scenario_type: 天气场景类型，用于调整不确定性
        params: 系统参数
        cost_params: 成本参数
        use_relaxed_binary: 是否松弛二进制变量为连续变量
        verbose: 是否打印求解过程

    Returns:
        DROResult: 优化结果
    """
    start_time = time.time()

    # 初始化参数
    if params is None:
        params = SystemParameters()
    if cost_params is None:
        cost_params = CostParameters()

    T = params.T

    # 提取输入数据
    v_wind = nominal_data['v_wind']
    I_DNI = nominal_data['I_DNI']
    electricity_price = nominal_data['price']
    L_tt = nominal_data['L_tt']

    # 计算单位功率
    P_w_i, P_pv_unit, P_pv_single_STC = calculate_unit_power(params, v_wind, I_DNI)

    # 根据场景类型调整不确定性半径
    if scenario_type is not None:
        uncertainty_factor = get_uncertainty_factor(scenario_type)
        effective_rho = rho * uncertainty_factor
        if verbose:
            print(f"场景类型: {scenario_type.value}, 不确定性因子: {uncertainty_factor}")
    else:
        effective_rho = rho

    # 获取每个时段的天气场景（用于协变量调整）
    weather_scenario = get_weather_scenario(I_DNI)

    # 计算每个时段的有效不确定性半径
    time_varying_rho = np.zeros(T)
    for t in range(T):
        if weather_scenario[t] == 1:  # 晴天
            time_varying_rho[t] = effective_rho * 0.5
        elif weather_scenario[t] == 2:  # 多云
            time_varying_rho[t] = effective_rho * 1.0
        else:  # 阴天/夜间
            time_varying_rho[t] = effective_rho * 1.5

    if verbose:
        print(f"有效不确定性半径范围: [{time_varying_rho.min():.3f}, {time_varying_rho.max():.3f}]")

    # ==================== 定义决策变量 ====================

    # 容量优化变量
    P_wind_capacity = cp.Variable(nonneg=True, name="P_wind_capacity")
    P_pv_capacity = cp.Variable(nonneg=True, name="P_pv_capacity")
    A_SF = cp.Variable(nonneg=True, name="A_SF")
    E_TES_max = cp.Variable(nonneg=True, name="E_TES_max")
    P_EH_Max = cp.Variable(nonneg=True, name="P_EH_Max")

    # 运行变量
    P_PB = cp.Variable(T, nonneg=True, name="P_PB")
    P_CSP = cp.Variable(T, nonneg=True, name="P_CSP")
    P_WE_WD = cp.Variable(T, nonneg=True, name="P_WE_WD")
    P_WE_PV = cp.Variable(T, nonneg=True, name="P_WE_PV")
    P_WC_WD = cp.Variable(T, nonneg=True, name="P_WC_WD")
    P_WC_PV = cp.Variable(T, nonneg=True, name="P_WC_PV")
    P_EH = cp.Variable(T, nonneg=True, name="P_EH")
    load_cut = cp.Variable(T, nonneg=True, name="load_cut")

    # 储热系统变量
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

    # 理论最大功率变量
    P_wd_max = cp.Variable(T, nonneg=True, name="P_wd_max")
    P_pv_max = cp.Variable(T, nonneg=True, name="P_pv_max")

    # 二进制/连续变量（用于储热充放电互斥）
    if use_relaxed_binary:
        x_TES_cha = cp.Variable(T, nonneg=True, name="x_TES_cha")
        x_TES_dis = cp.Variable(T, nonneg=True, name="x_TES_dis")
    else:
        x_TES_cha = cp.Variable(T, boolean=True, name="x_TES_cha")
        x_TES_dis = cp.Variable(T, boolean=True, name="x_TES_dis")

    # ==================== 鲁棒优化的对偶变量 ====================
    # 用于处理Box Uncertainty的对偶变换
    # 最坏情况下的额外成本变量
    lambda_wd = cp.Variable(T, nonneg=True, name="lambda_wd")  # 风电不确定性对偶
    lambda_pv = cp.Variable(T, nonneg=True, name="lambda_pv")  # 光伏不确定性对偶

    # ==================== 约束条件 ====================
    constraints = []

    # 【关键】容量边界约束（对应MATLAB second.txt 第245-249行）
    constraints.append(P_wind_capacity <= params.P_wind_max)  # 500 MW
    constraints.append(P_pv_capacity <= params.P_pv_max)  # 400 MW
    constraints.append(A_SF >= params.A_SF_min)  # 5000 m²
    constraints.append(A_SF <= params.A_SF_max)  # 5e6 m²
    constraints.append(E_TES_max <= params.E_TES_max_ub)  # 3000 MWh
    constraints.append(P_EH_Max <= params.P_EH_Max_ub)  # 200 MW

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

    # ==================== 鲁棒约束（Box Uncertainty） ====================
    # 风光出力在最坏情况下的约束
    # P_actual ∈ [P_nominal * (1 - rho), P_nominal * (1 + rho)]
    # 使用对偶变换处理min-max问题

    for t in range(T):
        rho_t = time_varying_rho[t]

        # 风电约束：在最坏情况下（出力最低时）仍需满足
        # P_WE_WD + P_WC_WD <= P_wd_max * (1 - rho_t)  （保守约束）
        constraints.append(P_WE_WD[t] + P_WC_WD[t] <= P_wd_max[t] * (1 - rho_t))

        # 光伏约束：在最坏情况下仍需满足
        constraints.append(P_WE_PV[t] + P_WC_PV[t] <= P_pv_max[t] * (1 - rho_t))

        # 对偶约束：捕获最坏情况的额外成本
        # 当实际出力高于调度时，产生弃风弃光成本
        constraints.append(lambda_wd[t] >= P_wd_max[t] * rho_t * cost_params.curtailment_penalty / 1e6)
        constraints.append(lambda_pv[t] >= P_pv_max[t] * rho_t * cost_params.curtailment_penalty / 1e6)

    # 热力循环回路能量平衡
    for t in range(T):
        constraints.append(Q_SF_HTF[t] + Q_TES_HTF[t] + Q_EH_HTF[t] == Q_HTF_TES[t] + Q_HTF_PB[t])

    # 负荷平衡约束
    for t in range(T):
        constraints.append(P_CSP[t] + P_WE_PV[t] + P_WE_WD[t] + load_cut[t] == L_tt[t])
        constraints.append(load_cut[t] <= L_tt[t])

    # 电加热只能使用弃风弃光量
    for t in range(T):
        constraints.append(P_EH[t] <= P_WC_WD[t] + P_WC_PV[t])

    # 弃风弃光吸纳率不低于90%
    constraints.append(cp.sum(P_EH) >= 0.9 * cp.sum(P_WC_WD + P_WC_PV))

    # 切负荷不超过总负荷的10%
    constraints.append(cp.sum(load_cut) <= 0.1 * np.sum(L_tt))

    # 储热系统约束
    for t in range(T):
        constraints.append(Q_TES_cha[t] == params.eta_TES_cha * Q_HTF_TES[t])
        constraints.append(Q_TES_dis[t] == Q_TES_HTF[t] / params.eta_TES_dis)

        # 储热容量约束
        constraints.append(E_TES[t] >= params.E_TES_min)
        constraints.append(E_TES[t] <= E_TES_max)

        # 储热系统动态方程
        if t == 0:
            constraints.append(E_TES[t] == params.E_init)
        else:
            constraints.append(E_TES[t] == (1 - params.gamma_TES) * E_TES[t - 1] +
                               (Q_TES_cha[t] - Q_TES_dis[t]) * params.Delta_t)

        # 储热系统充放热互斥
        constraints.append(x_TES_cha[t] + x_TES_dis[t] <= 1)
        if use_relaxed_binary:
            constraints.append(x_TES_cha[t] <= 1)
            constraints.append(x_TES_dis[t] <= 1)

        # 储热系统功率限值
        constraints.append(Q_HTF_TES[t] <= params.Q_HTF_TES_max * x_TES_cha[t])
        constraints.append(Q_TES_HTF[t] <= params.Q_TES_HTF_max * x_TES_dis[t])

    # 发电机组约束
    for t in range(T):
        # 热电转换
        constraints.append(P_PB[t] == params.eta_PB * Q_HTF_PB[t])

        # 出力约束
        constraints.append(P_PB[t] <= params.P_PB_Max)

        # 光热净发电
        constraints.append(P_CSP[t] == P_PB[t])

        # 电加热约束
        constraints.append(Q_EH_HTF[t] == params.eta_EH * P_EH[t])
        constraints.append(Q_EH_HTF[t] <= Q_HTF_TES[t] + Q_HTF_PB[t])
        constraints.append(P_EH[t] <= P_EH_Max)

    # 发电机组爬坡约束
    for t in range(1, T):
        constraints.append(P_PB[t] - P_PB[t - 1] <= params.Delta_P_Ru_PB)
        constraints.append(P_PB[t - 1] - P_PB[t] <= params.Delta_P_Rd_PB)

    # 太阳场约束
    for t in range(T):
        # 太阳场净热功率
        Q_SF_theoretical = params.eta_SF * I_DNI[t] * 1e-6  # MW/m²
        constraints.append(Q_SF_in[t] <= Q_SF_theoretical * A_SF - Q_Loss[t])
        constraints.append(Q_SF_HTF[t] == Q_SF_in[t] - Q_SF_cur[t])

    # 【关键修复】限制单日总发电量，防止Unbounded
    daily_total_generation = cp.sum(P_PB + P_WE_WD + P_WE_PV)
    constraints.append(daily_total_generation <= np.sum(L_tt) * 2.0)  # 不超过负荷的2倍

    # ==================== 目标函数 ====================

    # 投资成本
    cost_SF = cost_params.cost_SF_per_m2 * A_SF
    cost_TES = cost_params.cost_TES_per_kWh * 1000 * E_TES_max
    cost_PB = cost_params.cost_PB_per_kW * 1000 * params.P_PB_Max
    cost_PV = cost_params.cost_PV_per_kW * 1000 * P_pv_capacity
    cost_wind = cost_params.cost_wind_per_kW * 1000 * P_wind_capacity
    cost_EH = cost_params.cost_EH_per_kW * 1000 * P_EH_Max

    total_investment = cost_SF + cost_TES + cost_PB + cost_PV + cost_wind + cost_EH

    # 年运维成本
    OM_CSP = cost_params.OM_CSP_per_kW * 1000 * params.P_PB_Max
    OM_PV = cost_params.OM_PV_per_kW * 1000 * P_pv_capacity
    OM_wind = cost_params.OM_wind_per_kW * 1000 * P_wind_capacity
    mirror_maintenance = cost_params.mirror_maintenance * A_SF

    yearly_OM = OM_CSP + OM_PV + OM_wind + mirror_maintenance

    # 总成本（30年）
    total_cost = total_investment + yearly_OM * cost_params.project_lifetime

    # 日收益（名义场景）
    daily_revenue = cp.sum(cp.multiply(electricity_price, P_PB + P_WE_WD + P_WE_PV))

    # 惩罚项
    load_cut_penalty_val = cost_params.load_cut_penalty * cp.sum(load_cut)
    curtailment_penalty_val = cost_params.curtailment_penalty * cp.sum(P_WC_WD + P_WC_PV - P_EH)

    # 鲁棒优化的最坏情况额外成本
    worst_case_additional_cost = cp.sum(lambda_wd + lambda_pv) * 365 * cost_params.project_lifetime

    # 总收益（30年）
    total_revenue = daily_revenue * 365 * cost_params.project_lifetime

    # 目标：最大化利润 = 收益 - 成本 - 惩罚 - 最坏情况额外成本
    objective = (total_revenue
                 - total_cost
                 - load_cut_penalty_val * 365 * cost_params.project_lifetime
                 - curtailment_penalty_val * 365 * cost_params.project_lifetime
                 - worst_case_additional_cost)

    # ==================== 求解 ====================
    problem = cp.Problem(cp.Maximize(objective), constraints)

    solve_success = False

    # 定义求解器尝试列表
    solvers_list = []

    # 优先使用商业/高性能求解器（如果有）
    if 'GUROBI' in cp.installed_solvers():
        solvers_list.append(cp.GUROBI)
    if 'MOSEK' in cp.installed_solvers():
        solvers_list.append(cp.MOSEK)
    if 'CPLEX' in cp.installed_solvers():
        solvers_list.append(cp.CPLEX)

    # 其次使用开源鲁棒求解器
    # 1. 优先尝试 Clarabel (最适合云端的开源求解器)
    if 'CLARABEL' in cp.installed_solvers():
        solvers_list.append(cp.CLARABEL)

    # 2. 其次是 SCS
    if 'SCS' in cp.installed_solvers():
        solvers_list.append(cp.SCS)

    # 3. 最后尝试 ECOS (容易报错，放最后)
    if 'ECOS' in cp.installed_solvers():
        solvers_list.append(cp.ECOS)

    # 依次尝试
    for solver in solvers_list:
        try:
            if verbose:
                print(f"尝试求解器: {solver}")
            # 增加最大迭代次数，放宽精度要求以保证有解
            problem.solve(solver=solver, verbose=False, max_iters=10000)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                solve_success = True
                if verbose:
                    print(f"求解器 {solver} 成功，状态: {problem.status}")
                break
        except Exception as e:
            if verbose:
                print(f"求解器 {solver} 失败: {e}")
            continue

    # 如果都失败了，最后尝试不指定求解器（让CVXPY自己选）
    if not solve_success:
        try:
            if verbose:
                print("尝试默认求解器...")
            problem.solve(verbose=verbose)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                solve_success = True
        except Exception as e:
            if verbose:
                print(f"默认求解器也失败: {e}")

    solve_time = time.time() - start_time

    # ==================== 提取结果 ====================
    if problem.status in ['optimal', 'optimal_inaccurate']:
        # 安全提取变量值
        def safe_val(var, default=0):
            if var is None:
                return default
            if hasattr(var, 'value'):
                if var.value is None:
                    return default if not hasattr(var, 'shape') else np.zeros(var.shape)
                return var.value
            return var

        # 计算LCOE
        P_PB_val = safe_val(P_PB, np.zeros(T))
        P_WE_WD_val = safe_val(P_WE_WD, np.zeros(T))
        P_WE_PV_val = safe_val(P_WE_PV, np.zeros(T))

        annual_gen = np.sum(P_PB_val + P_WE_WD_val + P_WE_PV_val) * params.Delta_t * 365

        # 折现后的LCOE计算
        total_investment_val = safe_val(total_investment)
        yearly_OM_val = safe_val(yearly_OM)

        numerator = total_investment_val
        denominator = 0
        for year in range(1, cost_params.project_lifetime + 1):
            numerator += yearly_OM_val / (1 + cost_params.discount_rate) ** year
            denominator += annual_gen / (1 + cost_params.discount_rate) ** year

        LCOE = numerator / max(denominator, 1e-6) / 1000  # 转换为美元/kWh

        result = DROResult(
            P_wind_capacity=safe_val(P_wind_capacity),
            P_pv_capacity=safe_val(P_pv_capacity),
            A_SF=safe_val(A_SF),
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
            total_profit=problem.value if problem.value is not None else 0,
            worst_case_cost=safe_val(worst_case_additional_cost),
            LCOE=LCOE,
            status=problem.status,
            solve_time=solve_time
        )

        if verbose:
            print("\n" + "=" * 50)
            print("DRO优化结果")
            print("=" * 50)
            print(f"求解状态: {result.status}")
            print(f"求解时间: {result.solve_time:.2f} 秒")
            print(f"\n容量配置:")
            print(f"  风电容量: {result.P_wind_capacity:.2f} MW")
            print(f"  光伏容量: {result.P_pv_capacity:.2f} MW")
            print(f"  镜场面积: {result.A_SF:.0f} m²")
            print(f"  储热容量: {result.E_TES_max:.2f} MWh")
            print(f"  电加热容量: {result.P_EH_Max:.2f} MW")
            print(f"\n经济指标:")
            print(f"  总成本: {result.total_cost/1e6:.2f} 百万美元")
            print(f"  总收益: {result.total_revenue/1e6:.2f} 百万美元")
            print(f"  总利润: {result.total_profit/1e6:.2f} 百万美元")
            print(f"  LCOE: {result.LCOE*1000:.2f} 美元/MWh")
            print("=" * 50)

        return result
    else:
        if verbose:
            print(f"\n优化求解失败，状态: {problem.status}")

        # 返回空结果
        return DROResult(
            P_wind_capacity=0,
            P_pv_capacity=0,
            A_SF=0,
            E_TES_max=0,
            P_EH_Max=0,
            P_CSP=np.zeros(T),
            P_WE_WD=np.zeros(T),
            P_WE_PV=np.zeros(T),
            P_WC_WD=np.zeros(T),
            P_WC_PV=np.zeros(T),
            P_EH=np.zeros(T),
            E_TES=np.zeros(T),
            load_cut=np.zeros(T),
            total_cost=0,
            total_revenue=0,
            total_profit=0,
            worst_case_cost=0,
            LCOE=0,
            status=problem.status,
            solve_time=solve_time
        )


def run_sensitivity_analysis(
        nominal_data: Dict[str, np.ndarray],
        rho_values: list = None,
        params: Optional[SystemParameters] = None,
        cost_params: Optional[CostParameters] = None,
        verbose: bool = True
) -> Dict[str, list]:
    """
    运行不确定性半径的敏感性分析

    Args:
        nominal_data: 名义数据
        rho_values: 不确定性半径列表
        params: 系统参数
        cost_params: 成本参数
        verbose: 是否打印过程

    Returns:
        包含各指标随rho变化的字典
    """
    if rho_values is None:
        rho_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    results = {
        'rho': rho_values,
        'profit': [],
        'wind_capacity': [],
        'pv_capacity': [],
        'lcoe': [],
        'load_cut': [],
        'curtailment': []
    }

    for rho in rho_values:
        if verbose:
            print(f"\n正在求解 rho = {rho}...")

        result = solve_dro_model(
            nominal_data=nominal_data,
            rho=rho,
            params=params,
            cost_params=cost_params,
            verbose=False
        )

        if result.status in ['optimal', 'optimal_inaccurate']:
            results['profit'].append(result.total_profit / 1e6)
            results['wind_capacity'].append(result.P_wind_capacity)
            results['pv_capacity'].append(result.P_pv_capacity)
            results['lcoe'].append(result.LCOE * 1000)
            results['load_cut'].append(np.sum(result.load_cut))
            results['curtailment'].append(np.sum(result.P_WC_WD + result.P_WC_PV))

            if verbose:
                print(f"  利润: {result.total_profit/1e6:.2f} M$, "
                      f"风电: {result.P_wind_capacity:.1f} MW, "
                      f"光伏: {result.P_pv_capacity:.1f} MW")
        else:
            if verbose:
                print(f"  求解失败: {result.status}")
            # 填充NaN
            results['profit'].append(np.nan)
            results['wind_capacity'].append(np.nan)
            results['pv_capacity'].append(np.nan)
            results['lcoe'].append(np.nan)
            results['load_cut'].append(np.nan)
            results['curtailment'].append(np.nan)

    return results


def compare_scenarios(
        nominal_data: Dict[str, np.ndarray],
        rho: float = 0.1,
        params: Optional[SystemParameters] = None,
        cost_params: Optional[CostParameters] = None,
        verbose: bool = True
) -> Dict[str, DROResult]:
    """
    比较不同天气场景下的优化结果

    Args:
        nominal_data: 名义数据
        rho: 基础不确定性半径
        params: 系统参数
        cost_params: 成本参数
        verbose: 是否打印过程

    Returns:
        各场景的优化结果字典
    """
    scenarios = [ScenarioType.SUNNY, ScenarioType.CLOUDY, ScenarioType.RAINY]
    results = {}

    for scenario in scenarios:
        if verbose:
            print(f"\n正在求解场景: {scenario.value}...")

        result = solve_dro_model(
            nominal_data=nominal_data,
            rho=rho,
            scenario_type=scenario,
            params=params,
            cost_params=cost_params,
            verbose=False
        )

        results[scenario.value] = result

        if verbose and result.status in ['optimal', 'optimal_inaccurate']:
            print(f"  利润: {result.total_profit/1e6:.2f} M$")
            print(f"  风电容量: {result.P_wind_capacity:.1f} MW")
            print(f"  光伏容量: {result.P_pv_capacity:.1f} MW")

    return results


# 主程序入口
if __name__ == "__main__":
    print("=" * 60)
    print("分布鲁棒优化模型测试")
    print("=" * 60)

    # 获取默认数据
    data = get_default_data()

    # 测试基本求解
    print("\n1. 基本求解测试 (rho=0.1)")
    print("-" * 40)
    result = solve_dro_model(
        nominal_data=data,
        rho=0.1,
        verbose=True
    )

    if result.status in ['optimal', 'optimal_inaccurate']:
        print("\n测试通过!")
    else:
        print(f"\n测试失败，状态: {result.status}")

    # 测试敏感性分析
    print("\n2. 敏感性分析测试")
    print("-" * 40)
    sensitivity_results = run_sensitivity_analysis(
        nominal_data=data,
        rho_values=[0.05, 0.10, 0.15, 0.20],
        verbose=True
    )

    # 测试场景比较
    print("\n3. 场景比较测试")
    print("-" * 40)
    scenario_results = compare_scenarios(
        nominal_data=data,
        rho=0.1,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
