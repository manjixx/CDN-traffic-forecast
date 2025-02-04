import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# ========================
# 参数配置
# ========================
START_DATE = "2023-10-01 00:00:00"
END_DATE = "2023-10-07 23:55:00"
TIME_INTERVAL = "5T"  # 5分钟间隔
BASE_USERS_MEAN = 50000  # 平均基线用户数
PEAK_HOURS = [9, 10, 11, 14, 15, 16, 20, 21]  # 定义高峰时段

# ========================
# 生成时间序列
# ========================
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq=TIME_INTERVAL)
df = pd.DataFrame({"timestamp": date_range})

# ========================
# 模拟版本发布计划
# ========================
# 定义版本发布时间表
version_schedule = {
    "Android14_1.0": {
        "start_time": "2023-10-01 00:00:00",
        "batches": [
            {"batch_id": "Batch1", "duration_hours": 2, "user_ratio": 0.05},
            {"batch_id": "Batch2", "duration_hours": 6, "user_ratio": 0.20},
            {"batch_id": "Batch3", "duration_hours": 48, "user_ratio": 0.75}
        ],
        "update_size": 2.0
    },
    "iOS17_2.0": {
        "start_time": "2023-10-03 00:00:00",
        "batches": [
            {"batch_id": "Batch1", "duration_hours": 4, "user_ratio": 0.10},
            {"batch_id": "Batch2", "duration_hours": 12, "user_ratio": 0.30},
            {"batch_id": "Batch3", "duration_hours": 72, "user_ratio": 0.60}
        ],
        "update_size": 1.5
    }
}


# ========================
# 特征生成函数
# ========================
def generate_features(df, version_schedule):
    # 初始化字段
    df["version"] = "None"
    df["batch_id"] = "None"
    df["active_users"] = BASE_USERS_MEAN
    df["update_size_gb"] = 0.0
    df["update_ratio"] = 0.0
    df["retry_rate"] = 1.0
    df["is_peak"] = 0

    # 添加时间相关特征
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5

    # 模拟每个版本发布
    for ver, config in version_schedule.items():
        ver_start = pd.to_datetime(config["start_time"])
        update_size = config["update_size"]

        for batch in config["batches"]:
            batch_start = ver_start
            batch_end = batch_start + timedelta(hours=batch["duration_hours"])

            # 确定受影响的时段
            mask = (df["timestamp"] >= batch_start) & (df["timestamp"] < batch_end)

            # 更新版本信息
            df.loc[mask, "version"] = ver
            df.loc[mask, "batch_id"] = batch["batch_id"]
            df.loc[mask, "update_size_gb"] = update_size

            # 用户数叠加（基线+批次用户）
            batch_users = BASE_USERS_MEAN * batch["user_ratio"]
            time_decay = np.exp(-(df[mask]["timestamp"] - batch_start).dt.total_seconds() / (3600 * 24))
            df.loc[mask, "active_users"] += batch_users * time_decay

    # 生成动态特征
    # 1. 日内波动（正弦曲线模拟）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["active_users"] *= (1 + 0.3 * df["hour_sin"])  # ±30%波动

    # 2. 升级比例（时间相关）
    df["update_ratio"] = np.clip(
        0.1 + 0.3 * df["hour_sin"] + 0.1 * (df["hour"].isin(PEAK_HOURS)),
        0.05, 0.8
    )

    # 3. 重试率（高峰时段更高）
    df["retry_rate"] = np.where(
        df["hour"].isin(PEAK_HOURS),
        np.random.uniform(1.5, 2.0, len(df)),
        np.random.uniform(1.1, 1.3, len(df))
    )

    # 4. 添加随机噪声
    df["active_users"] *= np.random.uniform(0.95, 1.05, len(df))

    # 计算流量
    df["traffic_tb"] = (
            df["active_users"] * df["update_ratio"] *
            df["update_size_gb"] * df["retry_rate"] / 1e6  # 转换为TB
    ).round(2)

    # 标记95计费峰值（按小时统计前5%）
    hourly_peaks = df.set_index("timestamp").resample("H")["traffic_tb"].quantile(0.95)
    df["hourly_95"] = df["timestamp"].dt.floor("H").map(hourly_peaks)
    df["is_peak"] = (df["traffic_tb"] >= df["hourly_95"]).astype(int)

    return df.drop(columns=["hour_sin", "hourly_95"])


if __name__ == '__main__':

    df = generate_features(df, version_schedule)
    df.to_csv("../data/cdn_5min_traffic.csv", index=False)

    df = pd.read_csv("../data/cdn_5min_traffic.csv")


    # 绘制24小时流量趋势
    sample_day = df[(df["timestamp"] >= "2023-10-01") & (df["timestamp"] < "2023-10-02")]

    plt.figure(figsize=(14, 6))
    plt.plot(sample_day["timestamp"], sample_day["traffic_tb"], label="5分钟流量")
    # plt.plot(sample_day["timestamp"], sample_day["hourly_95"], 'r--', label="95计费阈值")
    plt.scatter(sample_day[sample_day["is_peak"]==1]["timestamp"],
                sample_day[sample_day["is_peak"]==1]["traffic_tb"],
                color='red', label="计费峰值点")
    plt.title("24小时流量趋势与95计费点")
    plt.xlabel("时间")
    plt.ylabel("流量 (TB)")
    plt.legend()
    plt.show()
