"""
CCG算法主程序 - 列约束生成算法
对应MATLAB文件: CCG.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import time

# 导入 MP 和 SP 求解器
from ccg_mp import solve_mp, solve_mp_start, ScheduleSolution, ScenarioData
from ccg_sp import solve_sp, solve_sp_simplified, BinaryVars, WorstCaseScenario, SPSolution


@dataclass
class CCGResult:
    """CCG算法完整结果"""
    converged: bool
    iterations: int
    final_LB: float
    final_UB: float
    gap: float
    gap_percentage: float
    LB_history: List[float]
    UB_history: List[float]
    gap_history: List[float]
    computation_time: float
    final_schedule: Any
    worst_case_scenarios: List[ScenarioData]
    final_solution_details: Dict[str, np.ndarray]


@dataclass
class CCGConfig:
    """CCG算法配置参数"""
    max_iterations: int = 10
    convergence_threshold: float = 1.0  # 绝对gap阈值（对应MATLAB: UB - LB <= 1）
    relative_gap_threshold: float = 0.01  # 相对gap阈值 1%
    use_simplified_sp: bool = True  # 使用简化版SP（枚举法）
    verbose: bool = True
    plot_results: bool = True


def run_ccg_algo(config: Optional[CCGConfig] = None) -> CCGResult:
    """
    运行列约束生成(CCG)算法

    对应MATLAB CCG.txt的主循环逻辑:
    1. 第一次迭代: MP_start() 使用名义场景
    2. 后续迭代: MP(UB, Load_u, Ppv_z, Pw_w) 使用最坏场景
    3. SP(Temp_net) 求解最坏场景
    4. 收敛条件: UB - LB <= 1
    """

    if config is None:
        config = CCGConfig()

    start_time = time.time()

    # ==================== 初始化（对应MATLAB CCG.txt 第4-5行） ====================
    # MATLAB: LB_recorder = -30000000; UB_recorder = +30000000;
    LB = -np.inf
    UB = np.inf

    LB_history = []
    UB_history = []
    gap_history = []

    # 基础数据（对应MATLAB CCG.txt 第8-18行）
    Load_base = np.array([53, 50, 48, 49, 61, 51, 56, 70, 64, 50,
                          45, 40, 18, 20, 30, 50, 71, 100, 84, 71,
                          60, 42, 24, 20])

    Ppv_base = 0.65 * np.array([0.00, 0.00, 0.00, 0.00, 0.00, 17.58, 43.65, 73.90, 93.48, 95.38,
                                102.30, 96.59, 101.40, 92.63, 93.50, 92.35, 68.09, 45.76, 21.81, 0.00,
                                0.00, 0.00, 0.00, 0.00])

    Pw_base = 0.65 * np.array([104.4555, 123.4790, 102.8142, 92.6088, 112.9052, 89.1061, 111.0100, 109.7712,
                               73.3869, 108.9974, 126.4295, 118.9272, 116.5876, 126.8384, 91.2581, 100.3876,
                               106.9537, 101.8204, 111.3992, 83.8046, 59.3542, 79.2348, 102.8312, 89.9582])

    # 存储最坏场景列表
    worst_case_scenarios: List[ScenarioData] = []

    # 当前最坏场景（用于传递给MP）
    current_Load_u = Load_base.copy()
    current_Ppv_z = Ppv_base.copy()
    current_Pw_w = Pw_base.copy()

    final_schedule = None
    final_solution_details = {}
    converged = False

    if config.verbose:
        print("=" * 60)
        print("列约束生成(CCG)算法")
        print("=" * 60)
        print(f"最大迭代次数: {config.max_iterations}")
        print(f"收敛阈值: {config.convergence_threshold}")
        print(f"使用简化SP: {config.use_simplified_sp}")
        print("=" * 60)

    # ==================== CCG主循环（对应MATLAB CCG.txt 第21-38行） ====================
    for iteration in range(1, config.max_iterations + 1):
        if config.verbose:
            print(f"\n--- 迭代 {iteration} ---")

        # -------------------- Step 1: 求解主问题 --------------------
        # 对应MATLAB CCG.txt 第23-27行
        if config.verbose:
            print("求解主问题 (MP)...")

        try:
            if iteration == 1:
                # 第一次迭代：使用MP_start（名义场景）
                # 对应MATLAB: [LB, Temp_net] = MP_start()
                LB_new, schedule = solve_mp_start()
            else:
                # 后续迭代：使用MP（传入最坏场景）
                # 对应MATLAB: [LB, Temp_net] = MP(UB, Load_u, Ppv_z, Pw_w)
                LB_new, schedule = solve_mp(
                    UB=UB,
                    Load_u=current_Load_u,
                    Ppv_z=current_Ppv_z,
                    Pw_w=current_Pw_w
                )

            if schedule is not None:
                LB = max(LB, LB_new)  # LB单调不减
                final_schedule = schedule
                if config.verbose:
                    print(f"  LB = {LB:.2f}")
            else:
                print(f"警告: 迭代{iteration}时MP不可行")
                # 尝试不加UB约束
                if iteration == 1:
                    LB_new, schedule = solve_mp_start()
                else:
                    LB_new, schedule = solve_mp(
                        UB=np.inf,
                        Load_u=current_Load_u,
                        Ppv_z=current_Ppv_z,
                        Pw_w=current_Pw_w
                    )
                if schedule is not None:
                    LB = max(LB, LB_new)
                    final_schedule = schedule
                else:
                    print(f"错误: 迭代{iteration}时MP求解失败")
                    break

        except Exception as e:
            print(f"MP求解错误 (迭代{iteration}): {e}")
            import traceback
            traceback.print_exc()
            break

        LB_history.append(LB)

        # -------------------- Step 2: 求解子问题 --------------------
        # 对应MATLAB CCG.txt 第30-31行
        # [UB, Load_u, Ppv_z, Pw_w, ...] = SP(Temp_net)
        if config.verbose:
            print("求解子问题 (SP)...")

        # 准备传递给SP的二进制变量
        binary_vars = BinaryVars(
            Temp_net=schedule.Temp_net,
            x_PB=schedule.x_PB,
            u_PB=schedule.u_PB,
            v_PB=schedule.v_PB,
            x_TES_cha=schedule.x_TES_cha,
            x_TES_dis=schedule.x_TES_dis
        )

        try:
            if config.use_simplified_sp:
                UB_new, worst_case = solve_sp_simplified(binary_vars)
                sp_solution = None
            else:
                UB_new, worst_case, sp_solution = solve_sp(binary_vars)

            if worst_case is not None:
                # 更新UB（单调不增）
                # 对应MATLAB: UB = min(UB_recorder(end), UB)
                UB = min(UB, UB_new)

                # 更新当前最坏场景（传递给下一轮MP）
                current_Load_u = worst_case.Load_u
                current_Ppv_z = worst_case.Ppv_z
                current_Pw_w = worst_case.Pw_w

                # 添加到场景列表
                new_scenario = ScenarioData(
                    Load_u=worst_case.Load_u.copy(),
                    Ppv_z=worst_case.Ppv_z.copy(),
                    Pw_w=worst_case.Pw_w.copy()
                )
                worst_case_scenarios.append(new_scenario)

                # 存储详细解（用于绘图）
                if sp_solution is not None:
                    final_solution_details = {
                        'P_CSP': sp_solution.P_CSP,
                        'E_TES': sp_solution.E_TES,
                        'Pbuy': sp_solution.Pbuy,
                        'Psell': sp_solution.Psell,
                        'P_DR': sp_solution.P_DR,
                        'P_WC_PV': sp_solution.P_WC_PV,
                        'P_WC_WD': sp_solution.P_WC_WD,
                        'P_EH': sp_solution.P_EH,
                        'Load_u': worst_case.Load_u,
                        'Ppv_z': worst_case.Ppv_z,
                        'Pw_w': worst_case.Pw_w
                    }
                else:
                    # 简化版SP也存储场景数据
                    final_solution_details = {
                        'Load_u': worst_case.Load_u,
                        'Ppv_z': worst_case.Ppv_z,
                        'Pw_w': worst_case.Pw_w,
                        'P_CSP': schedule.P_CSP if hasattr(schedule, 'P_CSP') else np.zeros(24),
                        'E_TES': schedule.E_TES if hasattr(schedule, 'E_TES') else np.zeros(24),
                        'Pbuy': schedule.Pbuy if hasattr(schedule, 'Pbuy') else np.zeros(24),
                        'Psell': schedule.Psell if hasattr(schedule, 'Psell') else np.zeros(24),
                        'P_DR': schedule.P_DR if hasattr(schedule, 'P_DR') else np.zeros(24),
                        'P_WC_PV': schedule.P_WC_PV if hasattr(schedule, 'P_WC_PV') else np.zeros(24),
                        'P_WC_WD': schedule.P_WC_WD if hasattr(schedule, 'P_WC_WD') else np.zeros(24),
                        'P_EH': schedule.P_EH if hasattr(schedule, 'P_EH') else np.zeros(24),
                    }

                if config.verbose:
                    print(f"  UB = {UB:.2f}")
            else:
                print(f"警告: 迭代{iteration}时SP未找到最坏场景")
                UB_history.append(UB)
                gap_history.append(UB - LB if np.isfinite(UB) and np.isfinite(LB) else np.inf)
                continue

        except Exception as e:
            print(f"SP求解错误 (迭代{iteration}): {e}")
            import traceback
            traceback.print_exc()
            UB_history.append(UB)
            gap_history.append(UB - LB if np.isfinite(UB) and np.isfinite(LB) else np.inf)
            continue

        UB_history.append(UB)

        # -------------------- Step 3: 检查收敛 --------------------
        # 对应MATLAB CCG.txt 第36-38行: if UB - LB <= 1, break
        gap = UB - LB
        gap_history.append(gap)

        # 计算相对gap
        if abs(UB) > 1e-6:
            relative_gap = gap / abs(UB)
        else:
            relative_gap = gap

        if config.verbose:
            print(f"  Gap = {gap:.2f} (相对: {relative_gap * 100:.2f}%)")
            print(f"  场景数: {len(worst_case_scenarios)}")

        # 检查收敛条件
        if gap <= config.convergence_threshold:
            if config.verbose:
                print(f"\n*** 收敛! 绝对gap {gap:.2f} <= {config.convergence_threshold} ***")
            converged = True
            break

        if relative_gap <= config.relative_gap_threshold:
            if config.verbose:
                print(f"\n*** 收敛! 相对gap {relative_gap * 100:.2f}% <= {config.relative_gap_threshold * 100}% ***")
            converged = True
            break

    # ==================== 结果整理 ====================
    computation_time = time.time() - start_time

    if not converged and config.verbose:
        print(f"\n*** 达到最大迭代次数 ({config.max_iterations})，未收敛 ***")

    # 创建结果对象
    result = CCGResult(
        converged=converged,
        iterations=len(LB_history),
        final_LB=LB,
        final_UB=UB,
        gap=UB - LB if np.isfinite(UB) and np.isfinite(LB) else np.inf,
        gap_percentage=(UB - LB) / abs(UB) * 100 if abs(UB) > 1e-6 else 0,
        LB_history=LB_history,
        UB_history=UB_history,
        gap_history=gap_history,
        computation_time=computation_time,
        final_schedule=final_schedule,
        worst_case_scenarios=worst_case_scenarios,
        final_solution_details=final_solution_details
    )

    if config.verbose:
        print("\n" + "=" * 60)
        print("CCG算法结果摘要")
        print("=" * 60)
        print(f"收敛: {converged}")
        print(f"迭代次数: {result.iterations}")
        print(f"最终LB: {LB:.2f}")
        print(f"最终UB: {UB:.2f}")
        print(f"Gap: {result.gap:.2f} ({result.gap_percentage:.2f}%)")
        print(f"计算时间: {computation_time:.2f} 秒")
        print(f"总场景数: {len(worst_case_scenarios)}")
        print("=" * 60)

    # 绘图
    if config.plot_results and len(LB_history) > 0:
        plot_ccg_results(result, Load_base, Ppv_base, Pw_base)

    return result


def plot_ccg_results(result: CCGResult, Load_base: np.ndarray,
                     Ppv_base: np.ndarray, Pw_base: np.ndarray):
    """绘制CCG算法结果图"""

    plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: 收敛曲线
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    iterations = range(1, len(result.LB_history) + 1)

    ax1.plot(iterations, result.LB_history, 'b-o', label='下界 (LB)', linewidth=2, markersize=8)
    ax1.plot(iterations, result.UB_history, 'r-s', label='上界 (UB)', linewidth=2, markersize=8)

    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('目标函数值 (元)', fontsize=12)
    ax1.set_title('CCG算法上下界收敛图', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(iterations))

    plt.tight_layout()
    plt.savefig('ccg_convergence.png', dpi=150)
    plt.show()

    # 图2: Gap收敛
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(iterations, result.gap_history, 'g-^', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', label='收敛阈值')

    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('Gap (UB - LB)', fontsize=12)
    ax2.set_title('最优性Gap收敛图', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(iterations))

    plt.tight_layout()
    plt.savefig('ccg_gap.png', dpi=150)
    plt.show()

    # 如果有详细解，绘制更多图
    if result.final_solution_details:
        details = result.final_solution_details
        hours = range(1, 25)

        # 图3: 负荷不确定性分析（对应MATLAB figure(2)）
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        Load_upper = Load_base * 1.1
        Load_lower = Load_base * 0.9

        ax3.fill_between(hours, Load_lower, Load_upper, alpha=0.3, color='blue', label='不确定性范围')
        ax3.plot(hours, Load_base, 'b--', linewidth=1.5, label='负荷预测值')
        ax3.plot(hours, details['Load_u'], 'r-', linewidth=2, label='负荷最坏场景')

        ax3.set_xlabel('时段', fontsize=12)
        ax3.set_ylabel('功率 (kW)', fontsize=12)
        ax3.set_title('负荷不确定性分析', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(list(hours))

        plt.tight_layout()
        plt.savefig('load_uncertainty.png', dpi=150)
        plt.show()

        # 图4: 光伏不确定性分析（对应MATLAB figure(3)）
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        Ppv_upper = Ppv_base * 1.1
        Ppv_lower = Ppv_base * 0.9

        ax4.fill_between(hours, Ppv_lower, Ppv_upper, alpha=0.3, color='orange', label='不确定性范围')
        ax4.plot(hours, Ppv_base, 'orange', linestyle='--', linewidth=1.5, label='光伏预测值')
        ax4.plot(hours, details['Ppv_z'], 'r-', linewidth=2, label='光伏最坏场景')

        ax4.set_xlabel('时段', fontsize=12)
        ax4.set_ylabel('功率 (kW)', fontsize=12)
        ax4.set_title('光伏不确定性分析', fontsize=14)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(list(hours))

        plt.tight_layout()
        plt.savefig('pv_uncertainty.png', dpi=150)
        plt.show()

        # 图5: 风电不确定性分析（对应MATLAB figure(4)）
        fig5, ax5 = plt.subplots(figsize=(12, 6))

        Pw_upper = Pw_base * 1.1
        Pw_lower = Pw_base * 0.9

        ax5.fill_between(hours, Pw_lower, Pw_upper, alpha=0.3, color='green', label='不确定性范围')
        ax5.plot(hours, Pw_base, 'g--', linewidth=1.5, label='风电预测值')
        ax5.plot(hours, details['Pw_w'], 'r-', linewidth=2, label='风电最坏场景')

        ax5.set_xlabel('时段', fontsize=12)
        ax5.set_ylabel('功率 (kW)', fontsize=12)
        ax5.set_title('风电不确定性分析', fontsize=14)
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(list(hours))

        plt.tight_layout()
        plt.savefig('wind_uncertainty.png', dpi=150)
        plt.show()

        # 图6: 分时电价（对应MATLAB figure(5)）
        fig6, ax6 = plt.subplots(figsize=(12, 5))

        lambda_price = np.array([300, 300, 300, 300, 300, 300, 300, 700, 700, 1300,
                                 1300, 1300, 1300, 1300, 700, 700, 1300, 1300, 1300, 1300,
                                 1300, 1300, 300, 300]) / 1000

        colors = ['green' if p < 0.5 else 'orange' if p < 1.0 else 'red' for p in lambda_price]
        ax6.bar(hours, lambda_price, color=colors, edgecolor='black', alpha=0.7)

        ax6.set_xlabel('时段', fontsize=12)
        ax6.set_ylabel('分时电价 (元/kWh)', fontsize=12)
        ax6.set_title('分时电价', fontsize=14)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticks(list(hours))

        plt.tight_layout()
        plt.savefig('electricity_price.png', dpi=150)
        plt.show()


# 主程序入口
if __name__ == "__main__":
    print("运行CCG算法测试...")

    config = CCGConfig(
        max_iterations=10,
        convergence_threshold=1.0,
        use_simplified_sp=True,  # 使用简化版SP
        verbose=True,
        plot_results=True
    )

    result = run_ccg_algo(config)

    print("\n测试完成!")