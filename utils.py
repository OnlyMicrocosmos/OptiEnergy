"""
数据加载辅助函数
从 all.txt 中提取硬编码数据并转换为 NumPy 数组
"""

import numpy as np
from typing import Dict


def load_default_data() -> Dict[str, np.ndarray]:
    """
    加载默认输入数据（对应 all.txt 中的硬编码数据）

    该函数提取 MATLAB 代码中的所有输入数据，并完成预处理计算。
    注意：MATLAB 代码中使用了系数调整（如 0.45 * 5），这里已完成计算。

    Returns:
        包含以下键的字典：
        - 'P_wd': 风电出力数据 (MW) - 已乘以系数 0.45 * 5
        - 'P_pv': 光伏出力数据 (MW) - 已乘以系数 0.45 * 5
        - 'I_DNI': 太阳直射辐射强度 (W/m²)
        - 'price': 电价数据 (元/MWh)
        - 'L_tt': 负荷需求数据 (MW) - 已乘以系数 0.7
        - 'L_st': 原始负荷需求数据 (MW)
        - 'v_wind': 风速数据 (m/s) - 用于其他模型
        - 'time_hours': 时间序列 (1-24)

    Example:
        >>> data = load_default_data()
        >>> print(f"风电出力范围: {data['P_wd'].min():.2f} - {data['P_wd'].max():.2f} MW")
        >>> print(f"负荷需求总量: {data['L_tt'].sum():.2f} MWh")
    """

    # ==================== 原始数据（来自 all.txt） ====================

    # 风电出力原始数据（MW）
    P_wd_1 = np.array([
        12.0689, 8.5662, 12.4262, 17.4022, 9.8139, 12.0633, 5.9847, 13.8371,
        10.4123, 4.9822, 4.2232, 6.2094, 11.3067, 33.0847, 21.6094, 12.0326,
        16.0073, 9.3553, 13.8792, 4.7155, 13.7928, 9.0076, 13.5399, 9.8148
    ])

    # 光伏出力原始数据（MW）
    P_pv_1 = np.array([
        0, 0, 0, 0, 0, 0, 0, 0.001184159, 0.220677134, 2.041627643,
        7.324570355, 32.28073353, 16.66810552, 21.19112019, 22.45754812,
        15.75532432, 18.56005547, 6.96306104, 1.598958202, 0.785685838,
        0.069879846, 0, 0, 0
    ])

    # 太阳直射辐射强度（W/m²）
    I_DNI = np.array([
        0, 0, 0, 0, 0, 0, 0, 0.0725, 4.5052, 59.4297,
        173.3273, 779.3091, 447.2624, 551.1209, 704.1094, 366.4278,
        531.1645, 236.8347, 89.0775, 57.2798, 5.2521, 0, 0, 0
    ])

    # 电价数据（元/MWh）
    price = np.array([
        300, 300, 300, 300, 300, 300, 300, 700, 700, 1300, 1300, 1300,
        1300, 1300, 700, 700, 1300, 1300, 1300, 1300, 1300, 1300, 300, 300
    ], dtype=float)

    # 原始负荷需求数据（MW）
    L_st = np.array([
        53, 50, 48, 49, 61, 51, 56, 70, 64, 50, 45, 40,
        18, 20, 30, 50, 71, 100, 84, 71, 60, 42, 24, 20
    ], dtype=float)

    # ==================== 数据预处理（应用系数） ====================

    # 风电出力：应用系数 0.45 * 5 = 2.25
    # MATLAB: P_wd = 0.45 * 5 * P_wd_1
    P_wd = 0.45 * 5 * P_wd_1

    # 光伏出力：应用系数 0.45 * 5 = 2.25
    # MATLAB: P_pv = 0.45 * 5 * P_pv_1
    P_pv = 0.45 * 5 * P_pv_1

    # 负荷需求：应用系数 0.7
    # MATLAB: L_tt = 0.7 * L_st
    L_tt = 0.7 * L_st

    # ==================== 生成辅助数据 ====================

    # 时间序列（1-24小时）
    time_hours = np.arange(1, 25)

    # 风速数据（用于其他模型，这里使用默认值）
    # 注意：all.txt 中没有风速数据，这里使用之前模型的默认值
    v_wind = np.array([
        6.47, 6.7, 6.57, 7.53, 7.33, 7.22, 7.4, 7.1, 6.82, 7.41,
        6.07, 6.16, 6.84, 7.01, 6.77, 7.83, 7.32, 7.86, 7.53, 7.29,
        6.59, 7.08, 6.61, 6.48
    ])

    # ==================== 构建返回字典 ====================

    data_dict = {
        # 主要输入数据（已预处理）
        'P_wd': P_wd,  # 风电出力 (MW)
        'P_pv': P_pv,  # 光伏出力 (MW)
        'I_DNI': I_DNI,  # 太阳辐射强度 (W/m²)
        'price': price,  # 电价 (元/MWh)
        'L_tt': L_tt,  # 负荷需求 (MW)

        # 原始数据（未处理）
        'P_wd_raw': P_wd_1,  # 原始风电数据
        'P_pv_raw': P_pv_1,  # 原始光伏数据
        'L_st': L_st,  # 原始负荷数据

        # 辅助数据
        'v_wind': v_wind,  # 风速数据 (m/s)
        'time_hours': time_hours,  # 时间序列

        # 数据统计信息（便于检查）
        'T': len(P_wd),  # 时间段数
        'scaling_factor_renewable': 0.45 * 5,  # 风光系数
        'scaling_factor_load': 0.7,  # 负荷系数
    }

    return data_dict


def load_system_parameters() -> Dict[str, float]:
    """
    加载系统参数（对应 all.txt 中的系统参数设置）

    Returns:
        包含所有系统参数的字典

    Example:
        >>> params = load_system_parameters()
        >>> print(f"储热容量: {params['E_TES_max']} MWh")
        >>> print(f"发电机组最大出力: {params['P_PB_Max']} MW")
    """

    params = {
        # 时间参数
        'T': 24,  # 时间段数
        'Delta_t': 1.0,  # 时间间隔（小时）

        # 光热系统参数
        'eta_SF': 0.4,  # 光热转化效率
        'A_SF': 225000.0,  # 太阳场有效集热面积 (m²)
        'gamma_TES': 0.031,  # 储热系统热损失系数
        'E_TES_max': 1800.0,  # 最大储热能量 (MWh)
        'E_TES_min': 30.0,  # 最小储热能量 (MWh)
        'E_init': 900.0,  # 初始储热能量 (MWh)
        'eta_TES_cha': 0.98,  # 储热充热效率
        'eta_TES_dis': 0.98,  # 储热放热效率
        'Q_HTF_TES_max': 150.0,  # 最大吸热功率 (MW)
        'Q_TES_HTF_max': 150.0,  # 最大放热功率 (MW)

        # 发电机组参数
        'eta_PB': 0.38,  # 热电转换效率
        'P_PB_Min': 0.0,  # 发电机组最小出力 (MW)
        'P_PB_Max': 80.0,  # 发电机组最大出力 (MW)
        'Delta_P_Ru_PB': 50.0,  # 上调速率 (MW/h)
        'Delta_P_Rd_PB': 50.0,  # 下调速率 (MW/h)
        'T_min_On_PB': 1,  # 最短开机时间 (小时)
        'T_min_Off_PB': 1,  # 最短停机时间 (小时)

        # 电加热器参数
        'eta_EH': 0.95,  # 电加热设备电转热效率
        'P_EH_Max': 80.0,  # 电加热最大功率 (MW)

        # 惩罚系数（从目标函数推断）
        'load_cut_penalty': 5000.0,  # 切负荷惩罚 (元/MWh)
        'curtailment_penalty': 300.0,  # 弃风弃光惩罚 (元/MWh)
    }

    return params


def print_data_summary(data: Dict[str, np.ndarray]):
    """
    打印数据摘要信息

    Args:
        data: load_default_data() 返回的数据字典

    Example:
        >>> data = load_default_data()
        >>> print_data_summary(data)
    """
    print("\n" + "=" * 70)
    print("数据加载摘要 (all.txt)")
    print("=" * 70)

    print("\n--- 基本信息 ---")
    print(f"时间段数: {data['T']}")
    print(f"风光系数: {data['scaling_factor_renewable']}")
    print(f"负荷系数: {data['scaling_factor_load']}")

    print("\n--- 风电数据 ---")
    print(f"原始数据范围: {data['P_wd_raw'].min():.2f} - {data['P_wd_raw'].max():.2f} MW")
    print(f"处理后范围: {data['P_wd'].min():.2f} - {data['P_wd'].max():.2f} MW")
    print(f"日总发电量: {data['P_wd'].sum():.2f} MWh")
    print(f"平均出力: {data['P_wd'].mean():.2f} MW")

    print("\n--- 光伏数据 ---")
    print(f"原始数据范围: {data['P_pv_raw'].min():.2f} - {data['P_pv_raw'].max():.2f} MW")
    print(f"处理后范围: {data['P_pv'].min():.2f} - {data['P_pv'].max():.2f} MW")
    print(f"日总发电量: {data['P_pv'].sum():.2f} MWh")
    print(f"峰值出力: {data['P_pv'].max():.2f} MW")
    print(f"峰值时段: {np.argmax(data['P_pv']) + 1} 时")

    print("\n--- 太阳辐射数据 ---")
    print(f"辐射强度范围: {data['I_DNI'].min():.2f} - {data['I_DNI'].max():.2f} W/m²")
    print(f"峰值辐射: {data['I_DNI'].max():.2f} W/m²")
    print(f"峰值时段: {np.argmax(data['I_DNI']) + 1} 时")
    print(f"日照时段数: {np.sum(data['I_DNI'] > 0)}")

    print("\n--- 电价数据 ---")
    print(f"电价范围: {data['price'].min():.0f} - {data['price'].max():.0f} 元/MWh")
    print(f"平均电价: {data['price'].mean():.2f} 元/MWh")
    high_price_hours = np.sum(data['price'] >= 1300)
    print(f"高电价时段数 (≥1300元): {high_price_hours}")

    print("\n--- 负荷数据 ---")
    print(f"原始负荷范围: {data['L_st'].min():.2f} - {data['L_st'].max():.2f} MW")
    print(f"调整后范围: {data['L_tt'].min():.2f} - {data['L_tt'].max():.2f} MW")
    print(f"日总负荷: {data['L_tt'].sum():.2f} MWh")
    print(f"峰值负荷: {data['L_tt'].max():.2f} MW")
    print(f"峰值时段: {np.argmax(data['L_tt']) + 1} 时")
    print(f"平均负荷: {data['L_tt'].mean():.2f} MW")

    print("\n--- 供需平衡初步分析 ---")
    total_renewable = data['P_wd'].sum() + data['P_pv'].sum()
    total_load = data['L_tt'].sum()
    print(f"日总可再生能源: {total_renewable:.2f} MWh")
    print(f"日总负荷需求: {total_load:.2f} MWh")
    print(f"可再生能源占比: {total_renewable / total_load * 100:.2f}%")

    if total_renewable >= total_load:
        print(f"可再生能源过剩: {total_renewable - total_load:.2f} MWh")
    else:
        print(f"需要光热补充: {total_load - total_renewable:.2f} MWh")

    print("\n" + "=" * 70)


def validate_data(data: Dict[str, np.ndarray]) -> bool:
    """
    验证数据的有效性

    Args:
        data: load_default_data() 返回的数据字典

    Returns:
        True 如果数据有效，否则 False

    Example:
        >>> data = load_default_data()
        >>> if validate_data(data):
        ...     print("数据验证通过")
    """
    print("\n--- 数据验证 ---")

    valid = True

    # 检查数组长度
    T = data['T']
    arrays_to_check = ['P_wd', 'P_pv', 'I_DNI', 'price', 'L_tt']

    for key in arrays_to_check:
        if len(data[key]) != T:
            print(f"❌ {key} 长度错误: 期望 {T}, 实际 {len(data[key])}")
            valid = False
        else:
            print(f"✓ {key} 长度正确: {T}")

    # 检查非负性
    for key in ['P_wd', 'P_pv', 'I_DNI', 'L_tt']:
        if np.any(data[key] < 0):
            print(f"❌ {key} 包含负值")
            valid = False
        else:
            print(f"✓ {key} 所有值非负")

    # 检查电价合理性
    if np.any(data['price'] <= 0):
        print(f"❌ 电价包含非正值")
        valid = False
    else:
        print(f"✓ 电价所有值为正")

    # 检查数据类型
    for key in arrays_to_check:
        if not isinstance(data[key], np.ndarray):
            print(f"❌ {key} 不是 NumPy 数组")
            valid = False

    if valid:
        print("\n✓ 所有数据验证通过")
    else:
        print("\n❌ 数据验证失败")

    return valid


def export_data_to_csv(data: Dict[str, np.ndarray], filename: str = "input_data.csv"):
    """
    将数据导出为 CSV 文件

    Args:
        data: load_default_data() 返回的数据字典
        filename: 输出文件名

    Example:
        >>> data = load_default_data()
        >>> export_data_to_csv(data, "my_data.csv")
    """
    import pandas as pd

    # 创建 DataFrame
    df = pd.DataFrame({
        '时段': data['time_hours'],
        '风电出力(MW)': data['P_wd'],
        '光伏出力(MW)': data['P_pv'],
        '太阳辐射(W/m²)': data['I_DNI'],
        '电价(元/MWh)': data['price'],
        '负荷需求(MW)': data['L_tt'],
        '原始负荷(MW)': data['L_st'],
    })

    # 保存为 CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n数据已导出到: {filename}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("=" * 70)
    print("数据加载模块测试")
    print("=" * 70)

    # 1. 加载数据
    print("\n>>> 加载默认数据...")
    data = load_default_data()

    # 2. 打印摘要
    print_data_summary(data)

    # 3. 验证数据
    is_valid = validate_data(data)

    # 4. 加载系统参数
    print("\n>>> 加载系统参数...")
    params = load_system_parameters()
    print(f"\n系统参数数量: {len(params)}")
    print("\n主要参数:")
    print(f"  储热容量: {params['E_TES_max']} MWh")
    print(f"  发电机组最大出力: {params['P_PB_Max']} MW")
    print(f"  电加热最大功率: {params['P_EH_Max']} MW")
    print(f"  光热转化效率: {params['eta_SF']}")

    # 5. 导出数据
    print("\n>>> 导出数据到 CSV...")
    export_data_to_csv(data)

    # 6. 数据访问示例
    print("\n>>> 数据访问示例:")
    print(f"第12时段数据:")
    t = 11  # 索引从0开始
    print(f"  风电出力: {data['P_wd'][t]:.2f} MW")
    print(f"  光伏出力: {data['P_pv'][t]:.2f} MW")
    print(f"  负荷需求: {data['L_tt'][t]:.2f} MW")
    print(f"  电价: {data['price'][t]:.0f} 元/MWh")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)