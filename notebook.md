# CDN 流量预测方案

## 方案概述

1. LSTM数据预处理流程
   - 数据预处理
   - 特征工程，特征增强：添加滞后特征，添加统计特征
   - 数据归一化：统一量纲、便于梯度的计算、加快收敛等
   - 构建时间序列样本：使用过去个n批次预测下一个批次,根据业务周期选择WINDOW_SIZE，例如：如果平均每周发布1个批次，窗口可设为7;如果版本发布间隔稳定，窗口设为批次间隔数
   - 数据集划分
2. 模型构建（LSTM、prophet、LSTM+prophet）
3. 模型训练与验证
4. 预测与反归一化

```text
原始数据 → 时间特征/分类编码 → 特征工程 → 归一化 → 滑动窗口 → 训练集/测试集 → LSTM输入
```

## 基本公式

CDN流量 = 活跃用户数 × 升级比例 × 升级包大小 × 平均下载次数


## 数据预处理

### 特征选择

#### case 1

- 用户基数：厂商的手机活跃用户数（例如：1000万～5000万）。

- 升级包大小：根据升级类型（大版本/小补丁），假设范围（例如：100MB～3GB）。

- 升级比例：每次OTA升级时，用户选择升级的比例（例如：30%～70%）。

- 下载次数：用户可能多次重试下载（例如：1.2～1.5次/用户）。
- CDN带宽

#### case 2

- 时间戳
- 目标版本号：厂商发布的版本号。 
- 用户当前版本号：用于区分批次
- 当前版本发布时间
- 当前批次发布时间
- 当前用户版本号的批次编号（batch_id）：同一版本的分批次编号（如 Batch1、Batch2）。
- 当前版本用户基数：厂商的手机活跃用户数（例如：1000万～5000万）。 
- 升级包大小：根据升级类型（大版本/小补丁），假设范围（例如：100MB～3GB）。 
- 升级比例：每次OTA升级时，用户选择升级的比例（例如：30%～70%）。 
- 下载次数：用户可能多次重试下载（例如：1.2～1.5次/用户）。
- 批次间隔：同一版本不同批次之间的时间间隔（例如3～7天）。
- 批次用户比例：每个批次覆盖的用户比例（例如第一批5%，第二批20%，第三批75%）。

### 特征工程

```python

"""
    特征增强
"""
# 添加滞后特征（过去1小时流量）
df_processed["traffic_lag12"] = df_processed["traffic_tb"].shift(12)  # 12*5min=1小时

# 添加统计特征
df_processed["traffic_ma6"] = df_processed["traffic_tb"].rolling(6).mean()  # 30分钟均线
df_processed["traffic_std6"] = df_processed["traffic_tb"].rolling(6).std()

# 处理时间周期性（使用正弦编码）
df_processed["hour_sin"] = np.sin(2 * np.pi * df_processed["hour"] / 24)
df_processed["hour_cos"] = np.cos(2 * np.pi * df_processed["hour"] / 24)

# 填充初始缺失值（因滚动计算产生）
df_processed = df_processed.dropna()
```

### 相关性分析

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 计算特征相关性矩阵
corr_matrix = df.corr()

# 可视化相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特征相关性矩阵")
plt.show()

# 提取与目标变量（traffic_tb）相关性较高的特征
target_corr = corr_matrix["traffic_tb"].sort_values(ascending=False)
print("目标变量相关性排序：\n", target_corr)

# 选择数值型特征列（排除分类变量和时间戳）
numerical_features = df.select_dtypes(include=[np.number]).columns.drop("timestamp")

# 计算Pearson相关系数矩阵
corr_matrix = df[numerical_features].corr()

# 绘制热力图
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("数值型特征相关性矩阵")
plt.show()

# 提取与目标变量的相关性排序
target_corr = corr_matrix["traffic_tb"].sort_values(ascending=False)
print("目标变量（traffic_tb）相关性排序：\n", target_corr)

# 使用方差分析（ANOVA）检验分类变量与流量的关系
from sklearn.feature_selection import f_classif

# 提取分类变量（已编码为One-Hot）
categorical_cols = [col for col in df.columns if "version_" in col or "batch_id_" in col]

# 计算F值和p值
f_values, p_values = f_classif(df[categorical_cols], df["traffic_tb"])

# 显示结果
categorical_corr = pd.DataFrame({
    "feature": categorical_cols,
    "f_value": f_values,
    "p_value": p_values
}).sort_values("f_value", ascending=False)

print("分类变量重要性：\n", categorical_corr.head(10))
```

### 数据归一化

针对不同类型数据处理方法不一致

- 分类变量
  - One-Hot编码：将分类变量转换为二进制向量
  - 嵌入层（Embedding Layer）：如果类别数量较多，可以使用嵌入层将高维稀疏向量映射到低维稠密向量
- 数值变量（如 update_size_gb）： 数值缩放到固定范围（如0~1）

### 备注

- 特征工程
  - 周期性编码：使用hour_sin/hour_cos替代原始时间特征，增强模型对周期性的捕捉
  - 滞后特征：引入traffic_lag12和traffic_ma6，帮助模型学习短期自相关性
  - 特征交互：尝试组合特征（如active_users * update_ratio）
  - 自动化特征工程：使用工具（如FeatureTools）生成更多候选特征,[自动化特征工程工具Featuretools应用](https://developer.aliyun.com/article/891368)
  - 不同版本发布策略，可以考虑使用embedding进行处理
   ![](https://pic3.zhimg.com/v2-7e0fbb2ba37a895c4986b4b8c66d7654_1440w.jpg)
  - 结合prophet模型
  
- 相关性分析
  - [方差分析详解](https://zhuanlan.zhihu.com/p/57896471)
    - F值高且p值低：表示该变量对流量影响显著 
    - F值低且p值高：表示该值可能对流量无显著影响

## 模型构建

### LSTM 模型

### prophet 模型

### LSTM + Prophet

#### 1. Prophet作为预处理工具（特征工程）
**核心思路**：
- **步骤1**：用Prophet分解时间序列的 **趋势（Trend）**、**季节性（Seasonality）** 和 **节假日效应（Holidays）**
- **步骤2**：将这些分解后的成分作为特征输入LSTM模型
- **步骤3**：LSTM专注于学习残差中的复杂模式

**实现步骤**：
```python
import pandas as pd
from fbprophet import Prophet

# 准备Prophet输入数据（需包含ds和y列）
df_prophet = df.rename(columns={"timestamp": "ds", "traffic_tb": "y"})

# 训练Prophet模型
model_prophet = Prophet(yearly_seasonality=False, daily_seasonality=True)
model_prophet.fit(df_prophet)

# 分解时序成分
components = model_prophet.predict(df_prophet)
trend = components["trend"]
daily_seasonality = components["daily"]  # 日内周期性

# 将Prophet输出作为特征添加到LSTM输入
df_lstm = pd.concat([
    df[["active_users", "update_size_gb", "retry_rate"]],  # 原始特征
    trend.rename("prophet_trend"),
    daily_seasonality.rename("prophet_daily_season"),
    df["traffic_tb"]  # 目标变量
], axis=1)

# 后续进行LSTM数据归一化和训练
```

**特点**：
- 显式捕获长期趋势和周期性，减轻LSTM学习负担
- Prophet分解精度影响最终效果
- 高频数据（5分钟）需要调整Prophet参数（如`changepoint_prior_scale`）

---

#### **2. Prophet与LSTM联合建模**

**核心思路**：
- **步骤1**：Prophet预测基础流量
- **步骤2**：LSTM预测流量残差（实际值 - Prophet预测值）
- **步骤3**：最终预测值 = Prophet预测 + LSTM残差预测

**实现步骤**：
```python
# Prophet预测
prophet_forecast = model_prophet.predict(df_prophet)["yhat"]

# 计算残差
df["residual"] = df["traffic_tb"] - prophet_forecast.values

# 训练LSTM预测残差
# （使用原始特征 + 时间特征作为输入）
X_train, y_train = create_sequences(df[["active_users", "hour_sin", "residual"]]) 

# 最终预测
final_pred = prophet_forecast + lstm_residual_pred
```

**优点**：
- 发挥Prophet在趋势外推和LSTM在残差学习的优势
- 适用于流量=基础趋势+随机波动的场景
- 依赖Prophet预测的准确性
- 需要维护两个模型

---

#### **3. 端到端混合模型（高级）**

**核心思路**：
- 将Prophet的季节性参数作为LSTM的初始化权重：如果目标是 **加速收敛** 且数据具有明显季节性，可以选择 **Prophet 参数初始化 LSTM**。
- 使用Attention机制融合Prophet输出：如果目标是 **灵活融合多源信息**，可以选择 **Attention 机制融合 Prophet**。

> **将Prophet的季节性参数作为LSTM的初始化权重**

**核心思想**
- **目标**：利用 Prophet 的季节性参数（如傅里叶级数系数）初始化 LSTM 的部分权重，使 LSTM 能够更快地学习到季节性模式。
- **方法**：
  1. 从 Prophet 模型中提取季节性参数（如傅里叶级数的正弦和余弦系数）。
  2. 将这些参数转换为 LSTM 的初始权重（如输入层或隐藏层的权重）。
  3. 在训练过程中，允许这些权重通过梯度下降进一步优化。

**实现步骤**
   1. **提取 Prophet 的季节性参数**：
      - Prophet 的季节性项通过傅里叶级数建模：
        ```
        seasonal = sum(a_k * sin(2πkt/P) + b_k * cos(2πkt/P))
        ```
        其中 `a_k` 和 `b_k` 是傅里叶系数，`P` 是周期长度。
      - 从 Prophet 模型中提取这些系数。

   2. **初始化 LSTM 权重**：
      - 将傅里叶系数映射到 LSTM 的输入层权重或隐藏层权重。
      - 例如，将 `a_k` 和 `b_k` 作为 LSTM 输入层的初始权重。

   3. **训练 LSTM**：在训练过程中，允许这些权重通过梯度下降优化。

**示例**
```python
import torch
import torch.nn as nn

# 假设从 Prophet 中提取的傅里叶系数
fourier_sin_coeffs = [0.1, 0.2, 0.3]  # 正弦项系数
fourier_cos_coeffs = [0.4, 0.5, 0.6]  # 余弦项系数

# 初始化 LSTM 输入层权重
input_size = len(fourier_sin_coeffs) + len(fourier_cos_coeffs)
hidden_size = 64

# 将傅里叶系数作为 LSTM 输入层的初始权重
lstm = nn.LSTM(input_size, hidden_size)
with torch.no_grad():
    lstm.weight_ih_l0[:len(fourier_sin_coeffs)] = torch.tensor(fourier_sin_coeffs)
    lstm.weight_ih_l0[len(fourier_sin_coeffs):] = torch.tensor(fourier_cos_coeffs)
```

**特点**
- **加速收敛**：LSTM 从 Prophet 的季节性参数开始学习，减少训练时间。
- **保留季节性模式**：确保 LSTM 能够捕捉到 Prophet 已经建模的季节性。
- **灵活性受限**：LSTM 的权重初始化依赖于 Prophet 的季节性参数，可能限制其捕捉其他模式的能力。
- **实现复杂**：需要从 Prophet 中提取参数并映射到 LSTM 的权重。

---

> **使用 Attention 机制融合 Prophet 输出**

**核心思想**
- **目标**：通过 Attention 机制动态加权 Prophet 的输出（如趋势项和季节性项）和 LSTM 的输出，实现更灵活的特征融合。
- **方法**：
  1. 将 Prophet 的输出（如 `trend` 和 `seasonal`）作为额外的特征输入到 LSTM 中。
  2. 使用 Attention 机制计算 Prophet 输出和 LSTM 输出的权重。
  3. 根据权重动态融合两者的输出。

**实现步骤**
1. **Prophet 输出作为特征**：
   - 将 Prophet 的 `trend` 和 `seasonal` 作为特征输入到 LSTM 中。

2. **Attention 机制**：
   - 使用 Attention 层计算 Prophet 输出和 LSTM 输出的权重。
   - 根据权重动态融合两者的输出。

3. **联合预测**：
   - 最终预测值 = Attention 加权后的 Prophet 输出 + Attention 加权后的 LSTM 输出。

**代码示例**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, lstm_out, prophet_out):
        # 拼接 LSTM 输出和 Prophet 输出
        combined = torch.cat([lstm_out, prophet_out], dim=-1)
        
        # 计算注意力权重
        weights = F.softmax(self.attention(combined), dim=-1)
        
        # 加权融合
        weighted_output = weights * lstm_out + (1 - weights) * prophet_out
        return weighted_output

class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # LSTM 层，输入维度为 input_size，输出维度为 hidden_size。
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 注意力层，用于融合 LSTM 和 Prophet 的输出。
        self.attention = Attention(hidden_size)
        # 全连接层，将融合后的特征映射到预测值。
        self.fc = nn.Linear(hidden_size, 1)
    # 前向传播
    # x：输入序列，形状为 (batch_size, seq_len, input_size)。
    # prophet_out：Prophet 模型的输出，形状为 (batch_size, hidden_size)。
    def forward(self, x, prophet_out):
        # x 输入序列
        lstm_out, _ = self.lstm(x)
        weighted_out = self.attention(lstm_out[:, -1, :], prophet_out)
        predictions = self.fc(weighted_out)
        return predictions
```

**特点**
- **动态融合**：Attention 机制能够根据数据动态调整 Prophet 和 LSTM 的贡献。
- **灵活性高**：适合复杂场景，能够捕捉 Prophet 和 LSTM 的互补信息。
- **计算成本高**：Attention 机制增加了模型复杂度。
- **实现复杂**：需要设计合理的 Attention 层和特征融合逻辑。

在使用LSTM模型预测连续值时，选择合适的考核指标至关重要，这些指标应能全面反映模型的预测精度、稳定性以及对时间序列特征的捕捉能力。以下是推荐的评估指标及其详细说明：

---

### **1. 基础回归指标**
#### **(1) 均方根误差（Root Mean Squared Error, RMSE）**
- **公式**：
  \[
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  \]
- **特点**：
  - 对较大误差敏感（平方项放大误差），适合关注极端值的场景。
  - 单位与原始数据一致，易于解释。
- **适用场景**：需惩罚大误差的预测任务，如流量峰值预测。

#### **(2) 平均绝对误差（Mean Absolute Error, MAE）**
- **公式**：
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **特点**：
  - 对异常值鲁棒，反映平均误差水平。
  - 无法区分误差方向（正负误差权重相同）。
- **适用场景**：需评估整体平均误差，如日常流量预测。

#### **(3) 平均绝对百分比误差（Mean Absolute Percentage Error, MAPE）**
- **公式**：
  \[
  \text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
  \]
- **特点**：
  - 以百分比形式表示误差，直观易懂。
  - 当真实值接近零时不稳定，且对负误差处理不友好。
- **适用场景**：数据无零值且需百分比误差分析，如增长率预测。

#### **(4) 决定系数（R² Score）**
- **公式**：
  \[
  R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
  \]
- **特点**：
  - 衡量模型对数据方差的解释能力，值越接近1表示拟合越好。
  - 无单位限制，适合模型间的横向对比。
- **适用场景**：评估模型相对于简单基准（如均值预测）的提升效果。

---

### **2. 时间序列特有指标**
#### **(1) 多步滚动预测误差**
- **方法**：分步预测未来多个时间点（如1步、3步、6步），分别计算各步长的RMSE/MAE。
- **意义**：评估模型在长期预测中的误差累积情况。
- **示例**：
  ```python
  def multi_step_metrics(y_true, y_pred, steps):
      metrics = {}
      for step in range(1, steps+1):
          rmse = np.sqrt(mean_squared_error(y_true[step-1::steps], y_pred[step-1::steps]))
          mae = mean_absolute_error(y_true[step-1::steps], y_pred[step-1::steps]))
          metrics[f"Step_{step}_RMSE"] = rmse
          metrics[f"Step_{step}_MAE"] = mae
      return metrics
  ```

#### **(2) 趋势捕捉能力**
- **方法**：计算预测序列与真实序列的趋势相关性（如皮尔逊相关系数）。
- **意义**：验证模型是否捕捉到数据的长期趋势。
- **公式**：
  \[
  \text{Trend Correlation} = \frac{\text{Cov}(y_{\text{trend}}, \hat{y}_{\text{trend}})}{\sigma_{y_{\text{trend}}} \sigma_{\hat{y}_{\text{trend}}}}
  \]

---

### **3. 业务相关指标**
#### **(1) 峰值预测准确率**
- **定义**：在流量峰值时间点（如前1%的高流量点），计算预测值与真实值的匹配度。
- **公式**：
  \[
  \text{Peak Accuracy} = \frac{1}{k} \sum_{i \in \text{Peaks}}} \left(1 - \frac{|y_i - \hat{y}_i|}{y_i}\right)
  \]
- **意义**：关键业务节点（如突发流量）的预测可靠性。

#### **(2) 误差分布分析**
- **方法**：绘制预测误差的直方图或箱线图，分析误差的偏态、峰态及异常值比例。
- **工具**：
  ```python
  import seaborn as sns
  errors = y_true - y_pred
  sns.histplot(errors, kde=True)  # 误差分布直方图
  sns.boxplot(x=errors)           # 误差箱线图
  ```

---

### **4. 模型稳定性评估**
#### **(1) 时间序列交叉验证（Time Series Split）**
- **方法**：按时间顺序划分多个训练-测试窗口，计算各窗口的指标均值和方差。
- **意义**：验证模型在不同时间段的表现稳定性。
- **示例**（使用`sklearn`）：
  ```python
  from sklearn.model_selection import TimeSeriesSplit
  tscv = TimeSeriesSplit(n_splits=5)
  for train_index, test_index in tscv.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      # 训练模型并计算指标
  ```

#### **(2) 滚动预测验证（Walk-Forward Validation）**
- **方法**：逐步扩展训练集，预测下一步并追加到训练数据中。
- **意义**：模拟实时预测场景，评估模型的持续适应能力。

---

### **5. 归一化指标（可选）**
- **归一化RMSE（Normalized RMSE）**：
  \[
  \text{NRMSE} = \frac{\text{RMSE}}{y_{\text{max}} - y_{\text{min}}}
  \]
- **归一化MAE（Normalized MAE）**：
  \[
  \text{NMAE} = \frac{\text{MAE}}{y_{\text{max}} \times 100\%
  \]
- **适用场景**：不同量纲数据集间的模型性能对比。

---

### **最终推荐指标组合**
| **指标类型**       | **推荐指标**                | **用途**                               |
|---------------------|-----------------------------|----------------------------------------|
| 基础精度评估       | RMSE, MAE, R²               | 整体预测精度与方差解释能力             |
| 时间序列适应性     | 多步滚动RMSE/MAE            | 长期预测误差分析                       |
| 业务关键点         | 峰值预测准确率              | 确保高流量时段的可靠性                 |
| 模型稳定性         | 时间序列交叉验证均值与方差  | 验证模型在不同时间段的鲁棒性           |

---

### **代码实现示例**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    # 基础指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE（确保无零值）
    if np.any(y_true == 0):
        mape = np.inf
    else:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 峰值准确率（取前1%作为峰值）
    peak_threshold = np.percentile(y_true, 99)
    peak_indices = np.where(y_true >= peak_threshold)
    peak_accuracy = 1 - np.mean(np.abs((y_true[peak_indices] - y_pred[peak_indices]) / y_true[peak_indices]))
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape,
        "Peak Accuracy": peak_accuracy
    }

# 使用示例
y_true = np.array([...])  # 真实值
y_pred = np.array([...])  # 预测值
metrics = evaluate_model(y_true, y_pred)
print(metrics)
```

---

### **总结**
- **核心指标**：RMSE（精度）、MAE（鲁棒性）、R²（模型解释力）。
- **时间序列特性**：多步滚动误差、趋势相关性。
- **业务适配**：峰值准确率、误差分布分析。
- **稳定性验证**：时间序列交叉验证、滚动预测。

通过综合这些指标，可以全面评估LSTM模型在连续值预测任务中的性能，确保模型既具备高精度，又能满足实际业务需求。


## 版本处理逻辑

```python
import pandas as pd

def calculate_metrics(file_path):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')  # 假设数据是用制表符分隔的
    
    # 计算每个时间点开放的版本包数量
    version_count = df.groupby('开放时间')['版本id'].nunique()
    
    # 计算每个时间点的开放在网量总数
    total_open_network = df.groupby('开放时间')['开放在网量'].sum()
    
    # 计算每个时间点的版本包大小平均值
    avg_package_size = df.groupby('开放时间')['版本包大小'].mean()
    
    # 计算每个版本 在网量 * 版本包大小 的和
    df['在网量_版本包大小'] = df['在网量'] * df['版本包大小']
    total_network_size = df.groupby('开放时间')['在网量_版本包大小'].sum()
    
    # 处理开放比例（根据开放尾号）
    def compute_open_ratio(row, prev_ratios):
        tail_num = str(row['开放尾号'])
        if tail_num.lower() == 'all':
            return 1 - prev_ratios.get(row['版本id'], 0)  # 计算剩余比例
        else:
            ratio = 10 ** (-len(tail_num))
            prev_ratios[row['版本id']] = prev_ratios.get(row['版本id'], 0) + ratio
            return ratio
    
    prev_ratios = {}  # 存储每个版本之前的开放比例
    df['计算开放比例'] = df.apply(lambda row: compute_open_ratio(row, prev_ratios), axis=1)
    
    # 返回计算结果
    return {
        '版本包数量': version_count,
        '开放在网量总数': total_open_network,
        '平均版本包大小': avg_package_size,
        '在网量_版本包大小之和': total_network_size,
        '计算开放比例': df[['版本id', '开放时间', '计算开放比例']]
    }

# 示例调用
# result = calculate_metrics("data.tsv")
# print(result)

```

```python
import pandas as pd

def parse_open_ratio(row, open_ratios):
    """
    解析开放尾号与开放比例的关系，并计算最终的开放比例。
    """
    if row['开放尾号'] == 'all':
        # 如果是 all，则用 1 减去该版本之前的总开放比例
        prev_open_ratio = open_ratios.get(row['版本号'], 0)
        return 1 - prev_open_ratio
    else:
        # 计算开放比例，尾号长度决定比例
        digits = len(str(row['开放尾号']))
        return 10 ** (-digits)

def calculate_metrics(file_path):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')  # 读取 TSV 格式的文件
    
    # 计算每个版本的累积开放比例
    open_ratios = {}
    df['最终开放比例'] = df.apply(lambda row: parse_open_ratio(row, open_ratios), axis=1)
    
    # 更新已开放的比例
    for index, row in df.iterrows():
        version = row['版本号']
        open_ratios[version] = open_ratios.get(version, 0) + row['最终开放比例']
    
    # 计算开放在网量
    df['开放在网量计算'] = df['最终开放比例'] * df['在网量']
    
    # 计算所需指标
    result = df.groupby('开放时间').agg(
        开放版本数量=('版本号', 'count'),
        开放在网量总数=('开放在网量计算', 'sum'),
        平均版本包大小=('版本包大小', 'mean'),
        在网量_版本包大小总和=('在网量', lambda x: (x * df.loc[x.index, '版本包大小']).sum())
    ).reset_index()
    
    print(result)
    return result

# 调用函数，传入数据集路径
# calculate_metrics("data.tsv")

```

```python
import pandas as pd

def parse_open_ratio(open_tail, previous_ratios):
    """
    根据开放尾号计算开放比例。
    - 一位数：10%
    - 两位数：1%
    - 三位数：0.1%
    - 'all'：剩余未开放的部分（1 - sum(之前的开放比例)）
    """
    if open_tail.lower() == 'all':
        return max(0, 1 - sum(previous_ratios))
    elif open_tail.isdigit():
        return len(open_tail) * (10 ** -len(open_tail))
    return 0  # 处理异常情况

def analyze_data(file_path):
    df = pd.read_csv(file_path, sep='\t')  # 读取 TSV 文件
    
    df['开放时间'] = pd.to_datetime(df['开放时间'])  # 转换时间格式
    df = df.sort_values(by='开放时间')  # 按开放时间排序
    
    result = {}
    version_ratios = {}  # 记录每个版本的累计开放比例
    
    for _, row in df.iterrows():
        time = row['开放时间']
        version = row['版本号']
        size = row['版本包大小']
        online_count = row['在网量']
        open_tail = str(row['开放尾号'])
        
        if version not in version_ratios:
            version_ratios[version] = []
        
        open_ratio = parse_open_ratio(open_tail, version_ratios[version])
        version_ratios[version].append(open_ratio)
        
        open_online_count = open_ratio * online_count  # 计算开放在网量
        weighted_online_size = open_online_count * size  # (在网量 * 版本包大小)
        
        if time not in result:
            result[time] = {'count': 0, 'total_open_online': 0, 'total_size': 0, 'weighted_size_sum': 0}
        
        result[time]['count'] += 1
        result[time]['total_open_online'] += open_online_count
        result[time]['total_size'] += size
        result[time]['weighted_size_sum'] += weighted_online_size
    
    for time in result:
        result[time]['avg_size'] = result[time]['total_size'] / result[time]['count'] if result[time]['count'] > 0 else 0
    
    # 转换为 DataFrame 方便查看
    result_df = pd.DataFrame.from_dict(result, orient='index')
    result_df.index.name = '开放时间'
    return result_df

# 示例调用（请替换为你的数据文件路径）
# df_result = analyze_data('data.tsv')
# print(df_result)

```


```python
# deepseek
import pandas as pd

def analyze_data(file_path):
    # 读取数据并添加列名
    df = pd.read_csv(file_path, sep='\t', header=None, names=[
        "索引", "编号", "版本id", "规则id", "默认开放策略",
        "版本号", "开放时间", "开放尾号", "升级策略",
        "版本包大小", "原版本号", "在网量", "开放比例", "开放在网量"
    ])
    
    # 转换时间格式并排序
    df['开放时间'] = pd.to_datetime(df['开放时间'], format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values(by='开放时间')

    # --- 关键改进1：向量化计算开放比例 ---
    def calculate_ratio(group):
        group = group.sort_values('开放时间')
        ratios = []
        cum_ratio = 0.0
        
        for _, row in group.iterrows():
            tail = str(row['开放尾号']).strip().lower()
            if tail == 'all':
                ratio = max(0.0, 1.0 - cum_ratio)
                cum_ratio = 1.0  # 标记已完全开放
            else:
                # 根据尾号位数计算比例
                ratio = 10 ** (-len(tail)) if tail.isdigit() else 0.0
                cum_ratio += ratio
            
            ratios.append(ratio)
        
        return pd.Series(ratios, index=group.index)

    # 按版本分组计算开放比例
    df['计算开放比例'] = df.groupby('版本号', group_keys=False).apply(calculate_ratio)

    # --- 关键改进2：高效聚合计算 ---
    # 计算每个时间点的指标
    result = df.groupby('开放时间').agg(
        版本包数量=('版本号', 'nunique'),
        开放在网量总数=('在网量', lambda x: (x * df.loc[x.index, '计算开放比例']).sum()),
        平均版本包大小=('版本包大小', 'mean'),
        在网量_版本包总和=('在网量', lambda x: (x * df.loc[x.index, '版本包大小'] * df.loc[x.index, '计算开放比例']).sum())
    )

    return result

# 示例调用
# result = analyze_data('data.tsv')
# print(result)
```
