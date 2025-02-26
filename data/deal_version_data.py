import pandas as pd
# def parse_open_ratio(open_tail, previous_ratios):
#     """
#     根据开放尾号计算开放比例。
#     - 一位数：10%
#     - 两位数：1%
#     - 三位数：0.1%
#     - 'all'：剩余未开放的部分（1 - sum(之前的开放比例)）
#     """
#     if open_tail.lower() == 'all':
#         return max(0, 1 - sum(previous_ratios))
#     elif open_tail.isdigit():
#         return len(open_tail) * (10 ** -len(open_tail))
#     return 0  # 处理异常情况



def parse_open_ratio(row, open_ratios):
    """
    解析开放尾号与开放比例的关系，并计算最终的开放比例。
    """
    blversion = row['blversion_number']
    source_version = row['sourceBlVersion_value']
    if row['number_openstrategy'] == 'all':
        # 如果是 all，则用 1 减去该版本之前的总开放比例
        prev_open_ratio = open_ratios.get((blversion, source_version), 0)
        print(f"当前版本: {blversion}, 源版本: {source_version}, 之前的开放比例: {prev_open_ratio}")
        return 1 - prev_open_ratio
    else:
        # 计算开放比例，尾号长度决定比例
        digits = len(str(row['number_openstrategy']))
        current_open_ratio = 10 ** (-digits)

        print(f"当前版本: {blversion}, 源版本: {source_version}, 尾号: {row['number_openstrategy']}, 当前开放比例: {current_open_ratio}")
        return current_open_ratio

def analyze_data(file_path):
    df = pd.read_csv(file_path)  # 读取 TSV 文件
    print(df.shape[0])
    df['start_time'] = pd.to_datetime(df['start_time'],format="%Y-%m-%d %H:%M:%S")# 转换时间格式
    df = df.sort_values(by='start_time')
    # 将秒位全部置为0
    df['start_time'] = df['start_time'].dt.floor('T')  # 'T' 表示分钟
    # 计算每个时间点的版本数
    unique_combinations = df.drop_duplicates(subset=['start_time','blversion_number', 'sourceBlVersion_value'])
    count = unique_combinations.groupby(['start_time']).size().reset_index(name='version_count')
    # 计算每条路径对应的开放比例
    df = df.merge(count, on='start_time', how='left')

    # 计算每个版本的累积开放比例
    open_ratios = {}
    df['最终开放比例'] = df.apply(lambda row: parse_open_ratio(row, open_ratios), axis=1)
    # # 更新已开放的比例
    for index, row in df.iterrows():
        version = (row['blversion_number'], row['sourceBlVersion_value'])
        open_ratios[version] = open_ratios.get(version, 0) + row['最终开放比例']

    # 将字典转换为 DataFrame
    df_open_ratios = pd.DataFrame(
        [(k[0], k[1], v) for k, v in open_ratios.items()],
        columns=['blversion_number', 'sourceBlVersion_value', 'open_ratio']
    )
    df_combined = pd.concat([df, df_open_ratios], ignore_index=True)
    df_combined.to_csv("test.csv")
    print(df_open_ratios)
    df.to_csv("version_data_1.csv", encoding='utf-8', index=False)



version_data = pd.read_csv('version.csv.csv')




if __name__ == '__main__':
    # 示例调用（请替换为你的数据文件路径）
    df_result = analyze_data('version.csv.csv')
    # print(df_result)
