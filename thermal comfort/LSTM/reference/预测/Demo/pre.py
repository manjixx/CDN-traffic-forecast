import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import data_processing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import LSTM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def predict(predict_data_path,result_path,model_path,feature_name,predict_length,hidden_size=64):
    """
    :param predict_data_path: 预测数据输入路径
    :param result_path: 预测结果保存路径
    :param model_path: 模型路径
    :param feature_name: 预测使用的特征名称
    :param predict_length: 预测序列长度
    :param hidden_size: 隐藏层大小
    """
    pass

feature_name=['供水压力', '回水压力', '供水温度', '功率', '气温', '回水温度']
data_pre_result_path = 'E:\Pycharm project\榆林\预测\Demo\test.xlsx'  # 预测结果保存路径
model_path = 'E:\Pycharm project\榆林\预测\Demo\model_4h.pt'  # 模型路径
input_size = len(feature_name)    # 输入特征数
num_layers = 2
sequence_length=24
predict_length=4
hidden_size=64

# 加载训练好的模型
feature_name=['供水压力', '回水压力', '供水温度', '功率', '气温', '回水温度']
data_pre_result_path = '.\test.xlsx'  # 预测结果保存路径
model_path = './model_24h.pt'  # 模型路径
input_size = len(feature_name)    # 输入特征数
num_layers = 2
sequence_length=24*3
predict_length=24
hidden_size=128

# 加载训练好的模型
model = LSTM(input_size, hidden_size, num_layers, output_size=predict_length)
model.load_state_dict(torch.load(model_path))
raw_data=pd.read_excel('./test.xlsx')
raw_data = raw_data[feature_name]

result=[]
for i in range(0,raw_data.shape[0]-sequence_length,predict_length):
    # 加载测试数据
    data = raw_data.iloc[i:i+sequence_length]
    # 标准化
    p_mean = data['功率'].mean()
    p_std = data['功率'].std()
    data = (data - data.mean()) / data.std()
    data = np.array(data)
    x = torch.from_numpy(data.astype(np.float32))
    x = torch.unsqueeze(x, dim=0)

    # 预测
    model.eval()
    with torch.no_grad():
        # 预测
        predict = model(x).flatten()
        # 逆标准化
        predict = predict * p_std + p_mean
        result+=predict.tolist()
        # # 储存
        # pre = pd.DataFrame(predict.numpy())
        # pre.to_excel(data_pre_result_path)

#计算MAPE
res=np.array(result)
actual=np.array(raw_data['功率'][sequence_length:])
MAPE=np.average(np.abs(actual-res)/actual)
print('MAPE:',MAPE)
#绘图
plt.figure(dpi=100)
plt.plot(res[:240],label='predict')
plt.plot(actual[:240],label='actual')
plt.legend()
plt.show()