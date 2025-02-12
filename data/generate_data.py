import pandas as pd
import numpy as np

# ==================== 生成时间序列 ====================
start_date = "2024-05-01"
end_date = "2025-01-31"
date_rng = pd.date_range(start=start_date, end=end_date, freq="5T")  # 5分钟间隔
print(f"总数据量：{len(date_rng)}条")

# ==================== 生成流量特征 ====================
# 基础参数
base_value = 5000  # 基础带宽值 (Mbps)
daily_amplitude = 2000  # 日波动幅度
weekly_amplitude = 1500  # 周波动幅度
noise_level = 500    # 随机噪声幅度
trend_growth = 0.0003  # 每小时增长系数

# 1. 长期趋势（随时间缓慢增长）
hours = np.arange(len(date_rng)) / 12  # 每5分钟=1/12小时
trend = base_value * (1 + trend_growth)**hours

# 2. 日周期（每日波动）
daily_cycle = daily_amplitude * np.sin(
    2 * np.pi * (date_rng.hour * 60 + date_rng.minute) / (24*60)
)

# 3. 周周期（周末流量增加）
weekday = date_rng.weekday  # 0=周一, 6=周日
# 周五/六流量增加，周一最低
weekly_cycle = weekly_amplitude * np.where(
    weekday >= 4,  # 周五到周日
    np.sin(2 * np.pi * (weekday - 4) / 3),
    -0.5 + 0.5 * np.sin(2 * np.pi * weekday / 5)
)

# 4. 随机噪声
noise = np.random.normal(0, noise_level, len(date_rng))

# 组合所有特征
total_bandwidth = trend + daily_cycle + weekly_cycle + noise
total_bandwidth = np.clip(total_bandwidth, 1000, None)  # 确保最小值

# ==================== 创建DataFrame ====================
df = pd.DataFrame({
    "time": date_rng.strftime("%Y-%m-%d %H:%M:%S"),
    "total_band_width": total_bandwidth.astype(int)
})

# ==================== 保存数据 ====================
df.to_csv("5min_bandwidth.csv", index=False)
print("数据已保存到 cdn_traffic.csv")
print(df.head(10))
