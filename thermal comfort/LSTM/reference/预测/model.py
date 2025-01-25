import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils import data_processing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层维度
        :param num_layers: lstm层层数
        :param output_size: 输出维度
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 如果使用Dataloader加载数据，batch_first参数设置True，输入输出的batchsize维度提前，与Dataloader打包的训练数据维度相同
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.predict = nn.Linear(hidden_size, output_size)
        self.fc1=nn.Linear(2*output_size,2*output_size)
        self.fc2=nn.Linear(2*output_size,output_size)
    def forward(self, x,future_feat=None):
        """
        :param x: 历史特征输入
        :param future_feat: 预测未来特征输入
        """
#         x = x.unsqueeze(0)
        # h0和c0默认值全为0
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #lstm输出三个tensor，分别为每个cell的输出与每层lstm最后一个cell的hn,cn，取第二层lstm最终状态的hn作为lstm层输出
        lstm_out, (hn,cn)= self.lstm(x)
        output = self.predict(hn[1])
        #如果使用了未来预报特征，历史特征经过lstm输出后与未来特征concat后接一层全连接输出
        if future_feat is not None:
            h=torch.concat([output,future_feat],axis=1)
            h1=F.relu(self.fc1(h))
            output=self.fc2(h1)
        return output

#训练函数
def train(train_data_path,model_path,feature_name,sequence_length,predict_length,max_epoch=50,hidden_size=64):
    """
    :param train_data_path: 训练数据路径
    :param model_path: 模型保存路径
    :param feature_name: 用于训练预测的特征名称
    :param sequence_length: 输入序列长度
    :param predict_length: 输出预测序列长度
    :param max_epoch: 训练轮数，默认50
    :param hidden_size: 隐藏层大小，默认64
    """
    input_size = len(feature_name)  # 输入特征数
    time_interval = predict_length  # 训练集时间窗滑动间隔
    num_layers = 2  # LSTM层层数
    batch_size = 32  # batchsize
    lr = 0.001  # 学习率

    # 加载训练数据
    data = pd.read_excel(train_data_path)
    data = data[feature_name]
    data_train = data_processing(data, sequence_length, predict_length, time_interval)
    X_train, X_test, y_train, y_test = train_test_split(data_train['data_in'], data_train['data_out'], test_size=0.2)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # 实例化网络
    model = LSTM(input_size, hidden_size, num_layers, output_size=predict_length)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    loss_func = nn.MSELoss()

    # train
    for epoch in tqdm(range(max_epoch)):
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save
    torch.save(model.state_dict(), model_path)

#预测函数
def predict(predict_data_path,result_path,model_path,feature_name,predict_length,hidden_size=64):
    """
    :param predict_data_path: 预测数据输入路径
    :param result_path: 预测结果保存路径
    :param model_path: 模型路径
    :param feature_name: 预测使用的特征名称
    :param predict_length: 预测序列长度
    :param hidden_size: 隐藏层大小
    """

    data_pre_path = predict_data_path  # 预测数据读取路径
    data_pre_result_path = result_path  # 预测结果保存路径
    model_path = model_path  # 模型路径
    input_size = len(feature_name)    # 输入特征数
    num_layers = 2

    # 加载训练好的模型
    model = LSTM(input_size, hidden_size, num_layers, output_size=predict_length)
    model.load_state_dict(torch.load(model_path))

    # 加载测试数据
    data = pd.read_excel(data_pre_path)
    data = data[feature_name]
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
        # 储存
        pre = pd.DataFrame(predict.numpy())
        pre.to_excel(data_pre_result_path)