import pandas as pd


def calculate_open_ratio(df):
    """
    计算每个升级路径的开放比例
    """
    # 预处理：确保按时间排序
    df = df.sort_values('start_time').reset_index(drop=True)

    # 初始化存储字典
    ratio_history = {}

    def get_ratio(row):
        key = (row['blversion_number'], row['sourceBlVersion_value'])

        if str(row['number_openstrategy']).lower() == 'all':
            # 获取历史累计比例
            history_sum = ratio_history.get(key, 0)
            ratio = max(0, 1 - history_sum)
            ratio_history[key] = 1  # 标记为已完成
            return ratio
        else:
            # 计算当前比例
            tail_len = len(str(row['number_openstrategy']))
            current_ratio = 10 ** (-tail_len)

            # 更新历史记录
            ratio_history[key] = ratio_history.get(key, 0) + current_ratio
            return current_ratio

    # 先创建 open_ratio 列
    df['open_ratio'] = df.apply(get_ratio, axis=1)
    return df


def analyze_upgrade_paths(file_path):
    # 读取数据并转换时间格式
    df = pd.read_csv(file_path)
    df['start_time'] = pd.to_datetime(df['start_time'])

    # 计算开放比例
    df = calculate_open_ratio(df)

    # 计算每个时间点的总在网量和总版本包大小
    df['total_device_in_network'] = df['open_ratio'] * df['device_cnt']
    df['total_package_size'] = df['open_ratio'] * df['device_cnt'] * df['totalSize_MB']

    # 先处理 blversion_number 和 sourceBlVersion_value 的组合
    df['version_pair'] = list(zip(df['blversion_number'], df['sourceBlVersion_value']))

    # 按时间点统计每个时间点的路径数量（升级路径数）和总在网量、总版本包大小
    time_group = df.groupby('start_time').agg(
        upgrade_path_count=('version_pair', 'nunique'),
        total_device_in_network=('total_device_in_network', 'sum'),
        total_package_size=('total_package_size', 'sum')
    ).reset_index()

    # 合并统计数据到原数据中
    df = pd.merge(df, time_group[['start_time', 'upgrade_path_count', 'total_device_in_network', 'total_package_size']],
                  on='start_time', how='left')

    return df


# 使用示例
df = analyze_upgrade_paths('version.csv')

# 输出结果
print("=== 更新后的原始数据 ===")
print(df)

# 保存为 CSV
df.to_csv('updated_version_data_with_statistics.csv', index=False)
