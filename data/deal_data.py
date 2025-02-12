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

# ==================== 保存为npy文件 ====================
# 组合特征矩阵（包含原始时间列和编码特征）
feature_columns = [
    "year", "month", "day", "hour", "minute",
    "day_sin", "day_cos", "hour_sin", "hour_cos",
    "minute_sin", "minute_cos", "total_band_width"
]
data_matrix = df[feature_columns].to_numpy()

# 保存为npy文件
np.save("cdn_traffic_processed.npy", data_matrix)
print("处理后的数据已保存至 cdn_traffic_processed.npy")

# ==================== 验证数据结构 ====================
print("\n数据结构示例：")
print(df[feature_columns].head(3))

print("\n数据维度：", data_matrix.shape)
