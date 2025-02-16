# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import LSTM, Dropout, Dense, GlobalMaxPooling1D
from keras import Sequential, callbacks
from tensorflow import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Corr_Analysis:

    def __init__(self):
        self.spearman_corr = 'spearman'
        self.kendall_corr = 'kendall'
        self.pearson_corr = 'pearson'

        self.feature_columns = [
            "time", "year", "month", "day", "hour", "minute",
            "day_sin", "day_cos", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "avg_5min", "avg_10min", "avg_15min", "avg_20min", "avg_30min", "avg_1h", "avg_2h",
            "total_band_width"
        ]

        self.target = 'total_band_width'
        self.lags = range(-6, 7)

    def npy_2_pd(self, data):
        df = pd.DataFrame(data, columns=self.feature_columns)
        # # 检查数据格式
        # print("数据前5行：")
        # print(df.head())
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
        return df

    def corr_analysis(self, data):
        """
        相关性分析-默认为 皮尔森系数
        :param data:
        :return:
        """
        df = self.npy_2_pd(data)

        corr_matrix = df.corr(self.pearson_corr)
        # 提取目标变量的相关性
        target_corr = corr_matrix[[self.target]].sort_values(
            by='total_band_width',
            ascending=False
        )
        # 输出相关性结果
        print("\n与total_bandwidth的相关性排序：")
        print(target_corr)

        # target_corr.plot(kind='barh', title='Feature Correlations')

        # 可视化相关性热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5
        )
        plt.title("特征相关性矩阵")
        plt.show()

    def calculate_p_values(self, data):
        """
        显著性检验
        :param data:
        :return:
        """
        df = self.npy_2_pd(data)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(df.mean())

        df_cols = pd.DataFrame(columns=df.columns)
        p_values = df_cols.transpose().join(df_cols, how='outer')
        print(p_values)
        for r in df.columns:
            for c in df.columns:
                p_values[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
        print("\nP值矩阵：")
        print(p_values)
        # 筛选显著相关特征（p<0.05）
        significant_features = p_values[(p_values[self.target] < 0.05)].index.tolist()
        print("\n显著相关特征（p<0.05）")
        print(significant_features)

    def shift_corr(self, data):
        df = self.npy_2_pd(data)
        target = self.target
        features = self.feature_columns[1:-1]
        shifted_corr = pd.DataFrame(index=self.lags, columns=features)

        # 确保数据为数值类型
        df = df.apply(pd.to_numeric, errors='coerce')

        # 删除全为NaN的列
        # df = df.dropna(axis=1, how='all')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(df.mean())

        for feature in features:
            for lag in self.lags:
                # 计算时移相关性
                if lag < 0:
                    # 负滞后：目标变量滞后
                    shifted_corr.loc[lag, feature] = df[target].shift(-lag).corr(df[feature])
                elif lag > 0:
                    # 正滞后：特征滞后
                    shifted_corr.loc[lag, feature] = df[target].corr(df[feature].shift(lag))
                else:
                    # 零滞后：原始相关性
                    shifted_corr.loc[lag, feature] = df[target].corr(df[feature])
            # 将结果转换为数值类型
                shifted_corr = shifted_corr.astype(float)

        # 可视化
        shifted_corr.T.plot(kind='line', marker='o', figsize=(14, 6))
        plt.title("时移相关性分析")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.show()

class Data_Processor:
    def __init__(self):
        # 数据划分比例
        self.train_all_ratio = 0.6  # 训练集占比
        self.val_all_ratio = 0.2  # 验证集占比
        self.test_all_ratio = 0.2  # 测试集占比（通常为 1 - train - val）

        # 时序参数
        self.continuous_sample_point_num = 24  # 连续样本点数
        self.forecast_step = 1  # 预测步长

        # 归一化器
        self.time_scaler = MinMaxScaler(feature_range=(0, 1))
        self.band_width_scaler = MinMaxScaler(feature_range=(0, 1))

        # 特征列定义
        self.select_columns = [4,5,6,7,8,9,10,11,18]
        self.time_indices = slice(1, 6)
        self.cyclic_indices = slice(6, 12)
        self.band_width_indices = slice(18, None)

    def make_set(self, data_set):
        """
        使用历史数据制作训练集和测试集
        :param data_set: 历史数据列表
        :return: train_set, test_set 归一化处理后的训练集合测试集
        """

        # 按照比例对数据进行分割
        data_size = len(data_set)
        train_size = int(data_size * self.train_all_ratio)
        val_size = int(data_size * self.val_all_ratio)
        train_set = data_set[:train_size, :]
        val_set = data_set[train_size: train_size + val_size, :]
        test_set = data_set[train_size + val_size:, :]

        print("data_set_shape:{}".format(data_set.shape))
        print("train_set_shape:{}".format(train_set.shape))
        print("test_set_shape:{}".format(test_set.shape))
        print("val_set_shape:{}".format(val_set.shape))

        # 对训练集和测试集进行归一化处理
        return self.normalization(train_set, val_set, test_set)

    def normalization(self, train_set, val_set, test_set):
        """对训练集合测试集进行归一化处理
        :param train_set: 未进行归一化处理的训练集数据
        :param val_set: 未进行归一化处理的验证集数据
        :param test_set: 未进行归一化的测试集数据
        :return: train_set, val_set, test_set 归一化处理后的训练集合测试集
        """

        # plt.figure(figsize=(14, 6))
        # plt.subplot(121)
        # plt.hist(train_set[:, self.band_width_indices], bins=30)
        # plt.title('unNormalized Bandwidth Features')
        # 验证切片是否正确
        feature = corr.feature_columns
        # 输出每个切片对应的列名
        time_columns = feature[self.time_indices]
        cyclic_columns = feature[self.cyclic_indices]
        band_width_columns = feature[self.band_width_indices]

        print("Time Columns:", time_columns)
        print("Cyclic Columns:", cyclic_columns)
        print("Bandwidth Columns:", band_width_columns)
        select_feature = []
        for i in self.select_columns:
            select_feature.append(feature[i])
        print("选择特征为：", select_feature)
        # 归一化时间特征
        train_set[:, self.time_indices] = self.time_scaler.fit_transform(train_set[:, self.time_indices])
        val_set[:, self.time_indices] = self.time_scaler.transform(val_set[:, self.time_indices])
        test_set[:, self.time_indices] = self.time_scaler.transform(test_set[:, self.time_indices])
        # 归一化流量带宽
        train_set[:, self.band_width_indices] = self.band_width_scaler.fit_transform(
            train_set[:, self.band_width_indices])
        val_set[:, self.band_width_indices] = self.band_width_scaler.transform(
            val_set[:, self.band_width_indices])
        test_set[:, self.band_width_indices] = self.band_width_scaler.transform(
            test_set[:, self.band_width_indices])

        print("周期特征范围验证:", np.max(train_set[:, self.cyclic_indices]), np.min(train_set[:, self.cyclic_indices]))

        # 2. 时间数值特征归一化到[0,1]
        print("时间特征范围验证:", np.max(train_set[:, self.time_indices]), np.min(train_set[:, self.time_indices]))
        print("带宽特征+标签范围验证:", np.max(train_set[:, -2:]), np.min(train_set[:, -2:]))

        # 3. 目标相关特征与标签保持相同分布
        #
        # plt.subplot(122)
        # plt.hist(train_set[:, self.band_width_indices], bins=30)
        # plt.title('Normalized Bandwidth Features')
        # plt.show()
        return train_set, val_set, test_set

    def time_inverse_normalization(self, data_set):
        """
        逆归一化
        :param data_set: 需要还原的数据
        :return:
        """
        # 对数据进行逆归一化还原
        print(data_set.shape)
        data_set[:, self.time_indices] = self.time_scaler.inverse_transform(data_set[:, self.time_indices])
        return data_set

    def target_inverse_normalization(self, data_set):
        """
        逆归一化
        :param data_set: 需要还原的数据
        :return:
        """
        # 对数据进行逆归一化还原
        data_set = self.band_width_scaler.inverse_transform(data_set)
        return data_set

    def create_sequence(self, data):
        """生成时间窗口样本，确保预测未来值
        :param data:输入数据
        """
        X, y = [], []
        for i in range(len(data) - self.continuous_sample_point_num - self.forecast_step + 1):
            X.append(data[i:i + self.continuous_sample_point_num, self.select_columns])  # 输入: [t-look_back, t)
            y.append(data[i + self.continuous_sample_point_num + self.forecast_step - 1, -1])  # 标签: t+forecast_step
        return np.array(X), np.array(y).flatten()

    def create_sequences(self, train_set, val_set, test_set):
        """创建所有序列
        :param train_set, val_set, test_set: 训练接，验证集，测试集
        :return: x_train_arr, y_train_arr, x_val_arr, y_val_arr, x_test_arr, y_test_arr
        """
        x_train, y_train = self.create_sequence(train_set)
        # 对训练集进行打乱
        seed = 7
        tf.random.set_seed(seed)
        x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=seed)

        # construct sequences for val_set
        x_val, y_val = self.create_sequence(val_set)

        # construct sequences for test_set
        x_test, y_test = self.create_sequence(test_set)

        print("x_train_shape:{}".format(x_train_shuffled.shape))
        print("y_train_shape:{}".format(y_train_shuffled.shape))
        print("x_val_shape:{}".format(x_val.shape))
        print("y_val_shape:{}".format(y_val.shape))
        print("x_test_shape:{}".format(x_test.shape))
        print("y_test_shape:{}".format(y_test.shape))
        return x_train_shuffled, y_train_shuffled, x_val, y_val, x_test, y_test


class CDN_LSTM_MODEL:
    def __init__(self, name):
        """
        构造函数，初始化模型
        :param data_list: 真实数据列表
        """
        # 神经网络名称
        self.name = name
        # 每次喂入神经网络的样本数
        self.batch_size = 64
        # 数据集的迭代次数
        self.epochs = 128
        # 每多少次训练集迭代，验证一次测试集
        self.validation_freq = 1
        # 配置模型
        self.model = Sequential([
            # LSTM层（记忆体个数，是否返回输出（True：每个时间步输出ht，False：仅最后时间步输出ht））
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(256, return_sequences=True),
            Dropout(0.2),
            # 配置具有100个记忆体的LSTM层，仅在最后一步返回ht
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            # LSTM(32, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            GlobalMaxPooling1D(),
            Dense(1)
        ])
        # 配置训练方法
        # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
        self.model.compile(
            optimizer=optimizers.legacy.Adam(0.001),
            loss='mean_squared_error',  # 损失函数用均方误差
            metrics='mean_absolute_error',
        )
        # 配置断点续训文件
        self.checkpoint_save_path = os.path.abspath(
            os.path.dirname(__file__)) + "/checkpoint/" + self.name + "_LSTM_cdn.ckpt"
        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('-' * 20 + "加载模型" + "-" * 20)
            self.model.load_weights(self.checkpoint_save_path)

        # 断点续训，存储最佳模型
        self.cp_callback = callbacks.ModelCheckpoint(filepath=self.checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor='val_loss',
                                                     )
        # 设定提前停止条件
        self.es_callback = callbacks.EarlyStopping(monitor='loss',
                                                   patience=5,
                                                   start_from_epoch=0
                                                   )

    def train(self, x_train, y_train, x_val, y_val):
        """
        训练模型
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        # 训练模型
        history = self.model.fit(x_train, y_train,
                                 # 每次喂入神经网络的样本数
                                 batch_size=self.batch_size,
                                 # 数据集的迭代次数
                                 epochs=self.epochs,
                                 validation_data=(x_val, y_val),
                                 # 每多少次训练集迭代，验证一次测试集
                                 validation_freq=self.validation_freq,
                                 callbacks=[self.cp_callback, self.es_callback])
        # 输出模型各层的参数状况
        self.model.summary()
        # 参数提取
        self.save_args_to_file()

        # 获取模型当前loss值
        loss = history.history['loss']
        print("loss:{}".format(loss))
        try:
            val_loss = history.history['val_loss']
            print("val_loss:{}".format(val_loss))
        except:
            pass

        plt.figure(figsize=(14, 6))
        plt.plot(history.history['loss'], c='b', label='loss')
        plt.plot(history.history['val_loss'], c='g', label='val_loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_args_to_file(self):
        """
        参数提取，将参数保存至文件
        :return:
        """
        # 指定参数存取目录
        file_path = os.path.abspath(
            os.path.dirname(__file__)) + "\\weights\\"
        # 目录不存在则创建
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # 打开文本文件
        file = open(file_path + self.name + "_weights.txt", 'w')
        # 将参数写入文件
        for v in self.model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()

    def test(self, x_test, test_set):
        """
        预测测试
        :param x_test:
        :param test_set: 测试集
        :return:
        """

        # 测试集输入模型进行预测
        predicted_cdn_traffic = self.model.predict(x_test)
        print('predicted_cdn_traffic {}'.format(predicted_cdn_traffic))

        valid_length = len(test_set) - processor.continuous_sample_point_num - processor.forecast_step + 1
        predict_bandwidth = np.copy(test_set[:valid_length, processor.band_width_indices])
        print('before replace predict_bandwidth[:, -1] {}'.format(predict_bandwidth[:, -1]))

        predict_bandwidth[:, -1] = predicted_cdn_traffic.flatten()

        print('after replace predict_bandwidth[:, -1] {}'.format(predict_bandwidth[:, -1]))

        # 对预测数据还原---从（0，1）反归一化到原始范围
        predict_bandwidth = processor.target_inverse_normalization(predict_bandwidth)
        predict_value = predict_bandwidth[:, -1].flatten()

        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_bandwidth = np.copy(test_set[:valid_length, processor.band_width_indices])
        print('real_bandwidth[:, -1] {}'.format(real_bandwidth[:, -1]))

        real_bandwidth = processor.target_inverse_normalization(real_bandwidth)
        real_value = real_bandwidth[:, -1].flatten()
        print('real_value {}'.format(real_value))
        print('predict_value {}'.format(predict_value))

        # evaluate
        # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
        mse = mean_squared_error(predict_value, real_value)
        # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
        rmse = math.sqrt(mean_squared_error(predict_value, real_value))
        # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
        mae = mean_absolute_error(predict_value, real_value)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)

        plt.figure(figsize=(14, 6))

        plt.plot(real_value, c='b', label='real_value')

        plt.plot(predict_value, c='g', label='predict_value')

        plt.tight_layout()

        plt.legend()

        plt.show()

    def predict(self, data):
        """
        使用模型进行预测
        :param history_data: 历史数据list
        :return:预测值
        """

        # 测试集输入模型进行预测
        predicted_cdn_traffic = self.model.predict(data)
        # 对预测数据还原---从（0，1）反归一化到原始范围
        predicted_cdn_traffic = self.denormalization(predicted_cdn_traffic)
        # 预测值
        value = predicted_cdn_traffic[-1][-1]
        print("预测值：{}".format(value))
        return value


if __name__ == '__main__':
    data = np.load("../data/5min_bandwidth_with_avg.npy", allow_pickle=True)

    # 查看每一列的数据类型
    corr = Corr_Analysis()
    # corr.corr_analysis(data)
    # corr.calculate_p_values(data)
    # corr.shift_corr(data)
    processor = Data_Processor()

    train_set, val_set, test_set = processor.make_set(data_set=data)
    #
    x_train, y_train, x_val, y_val, x_test, y_test = processor.create_sequences(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set
    )

    x_train = x_train.astype(float)
    x_val = x_val.astype(float)
    x_test = x_test.astype(float)

    print(y_train)  # 检查 NaN 数量
    print(y_val)  # 检查 inf 数量

    # # 初始化模型
    model = CDN_LSTM_MODEL(name="流量预测_V0.2.1")

    # 训练模型
    model.train(x_train, y_train, x_val, y_val)
    # 对模型进行测试
    model.test(x_test, test_set)

    # 利用模型进行预测
    # model.predict(x_test)
