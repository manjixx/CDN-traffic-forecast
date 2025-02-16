import pandas as pd
import numpy as np
from calendar import monthrange

# ==================== 读取数据 ====================
df = pd.read_csv("5min_bandwidth.csv")
df["time"] = pd.to_datetime(df["time"])

# ==================== 生成时间特征列 ====================
# 提取年月日时分秒
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day
df["hour"] = df["time"].dt.hour
df["minute"] = df["time"].dt.minute
df["second"] = df["time"].dt.second

# ==================== 周期性编码 ====================
# 1. 日周期（考虑当月实际天数）
df["days_in_month"] = df["time"].apply(lambda x: monthrange(x.year, x.month)[1])
df["day_sin"] = np.sin(2 * np.pi * (df["day"] - 1) / df["days_in_month"])
df["day_cos"] = np.cos(2 * np.pi * (df["day"] - 1) / df["days_in_month"])

# 2. 小时周期（24小时制）
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 3. 分钟周期（60分钟制）
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

# ==================== 清理冗余列 ====================
df.drop(columns=["days_in_month", "second"], inplace=True)  # 秒列全为0

# ==================== 计算带宽均值 ====================

# 确保按时间排序
df = df.sort_values("time").set_index("time")

# 检查时间间隔是否严格为5分钟
time_diff = df.index.to_series().diff().value_counts()
assert pd.Timedelta("5T") in time_diff, "数据时间间隔不符合5分钟要求"

# 定义需要计算的窗口列表
windows = {
    "5min": "5T",
    "10min": "10T",
    "15min": "15T",
    "20min": "20T",
    "30min": "30T",
    "1h": "1H",
    "2h": "2H"
}

# 计算滚动平均值（包含当前时间点之前的窗口）
for col_name, window_size in windows.items():
    df[f"avg_{col_name}"] = df["total_band_width"].rolling(
        window=window_size,
        closed="left",       # 包含窗口起点，不包含终点
        min_periods=1        # 允许最小1个数据点计算
    ).mean()

df = df.reset_index()

# ==================== 保存为npy文件 ====================
# 组合特征矩阵（包含原始时间列和编码特征）
feature_columns = [
            "time", "year", "month", "day", "hour", "minute",
            "day_sin", "day_cos", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "avg_5min", "avg_10min", "avg_15min", "avg_20min", "avg_30min", "avg_1h", "avg_2h",
            "total_band_width"
        ]

columns_to_fill = ["avg_5min", "avg_10min", "avg_15min", "avg_20min", "avg_30min", "avg_1h", "avg_2h"]

for col in columns_to_fill:
    df.loc[0, col] = df.loc[0, "total_band_width"]

data_matrix = df[feature_columns].to_numpy()

# 保存为npy文件
# np.save("cdn_traffic_processed.npy", data_matrix)
np.save("5min_bandwidth_with_avg.npy", data_matrix)
df.to_csv("5min_bandwidth_with_avg.csv")

print("处理后的数据已保存至 cdn_traffic_processed.npy")

# ==================== 验证数据结构 ====================
print("\n数据结构示例：")
print(df[feature_columns].head(3))

print("\n数据维度：", data_matrix.shape)
