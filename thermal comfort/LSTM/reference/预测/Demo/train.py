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
"""该脚本使用gpu训练"""

data1=pd.read_excel('./北站_train.xlsx').drop(['时间','Unnamed: 0'],axis=1)
data2=pd.read_excel('./南站_train.xlsx').drop(['时间','Unnamed: 0'],axis=1)
data1 = (data1 - data1.mean()) / data1.std()
data2 = (data2 - data2.mean()) / data2.std()
data=pd.concat((data1,data2)).reset_index(drop=True)
feature_name = ['供水压力', '回水压力', '供水温度', '功率', '气温', '回水温度']
model_path = './model_24h.pt'
sequence_length = 24*3
predict_length = 24
max_epoch =400
hidden_size = 128
# 加载训练数据
data = data[feature_name]

input_size = len(feature_name)  # 输入特征数
time_interval = predict_length  # 训练集时间窗滑动间隔
num_layers = 2  # LSTM层层数
batch_size = 64  # batchsize
lr = 0.001  # 学习率

data_train = data_processing(data, sequence_length, predict_length, time_interval)
X_train = torch.from_numpy(data_train['data_in'].astype(np.float32))
y_train = torch.from_numpy(data_train['data_out'].astype(np.float32))
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# 实例化网络
model = LSTM(input_size, hidden_size, num_layers, output_size=predict_length).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)
loss_func = nn.MSELoss().cuda()
train_loss_all = []
# train
for epoch in range(max_epoch):
    train_loss = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.cuda(), b_y.cuda()
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    print(f'Epoch{epoch + 1}/{max_epoch}: Loss:{train_loss / train_num}')
    train_loss_all.append(train_loss / train_num)

# save
torch.save(model.state_dict(), model_path)

#绘制loss
plt.figure(dpi=100)
plt.plot(train_loss_all)
plt.show()