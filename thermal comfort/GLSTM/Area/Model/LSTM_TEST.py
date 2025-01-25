# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

person_feature = ['male', 'female',
                  'young', 'old',
                  'short', 'medium', 'tall',
                  'thin', 'normal', 'fat',
                  'bmi_l', 'bmi_n', 'bmi_h',
                  'grf_l', 'grf_n', 'grf_h',
                  'sen_l', 'sen_n', 'sen_h',
                  'pre_l', 'pre_n', 'pre_h',
                  'env_l', 'env_n', 'env_h']
env_feature = ['date', 'time', 'season', 'va', 'ta', 'hr']
diff_feature = ['ta_diff1', 'ta_diff2']
avg_feature = ['height_avg', 'weight_avg', 'bmi_avg']
griffith = 'griffith_avg'
count_feature = 'count'
y_feature = 'tsv'


def data_loader():
    """
    'male', 'female','young', 'old','short', 'medium', 'tall','thin', 'normal', 'fat',
    'bmi_l', 'bmi_n', 'bmi_h',  'grf_l', 'grf_n', 'grf_h','sen_l', 'sen_n', 'sen_h',
    'pre_l', 'pre_n', 'pre_h', 'env_l', 'env_n', 'env_h'
    """
    person = np.load('../SDataSet/npy/' + filepath + '/person.npy', allow_pickle=True).astype(float)
    # 'date', 'time', 'season', 'va', 'ta', 'hr'
    env = np.load('../SDataSet/npy/' + filepath + '/env.npy', allow_pickle=True)
    org = []
    for i in range(0, round(len(env) / 4)):
        start = i * 4
        org.append(env[start: start + 1, 4:6])
    ta = env[:, 4:6]
    va = env[:, 3:4]

    season = env[:, 2:3]

    # ta_diff1, ta_diff2
    diff = np.load('../SDataSet/npy/' + filepath + '/diff.npy', allow_pickle=True).astype(float)
    # 'age_avg', 'height_avg', 'weight_avg', 'bmi_avg'
    avg = np.load('../SDataSet/npy/' + filepath + '/avg.npy', allow_pickle=True).astype(float)
    # griffith_avg
    griffith = np.load('../SDataSet/npy/' + filepath + '/grf.npy', allow_pickle=True).astype(float)
    # count
    count = np.load('../SDataSet/npy/' + filepath + '/count.npy', allow_pickle=True)

    # normalization: ['va', 'ta', 'hr', 'height_avg', 'weight_avg', 'bmi_avg'] env, avg
    normalization = np.concatenate((ta, avg), axis=1)

    normalization = scaler.fit_transform(normalization)
    # count, person, griffith_avg, season, diff, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg

    x = np.concatenate((count[:, None], person, griffith[:, None], season, diff, va, normalization), axis=1)

    x_split = []
    for i in range(0, round(len(x) / 4)):
        start = i * 4
        end = (i + 1) * 4
        x_hat = x[start: end, :]
        x_split.append(x_hat)

    x_split = tf.cast(x_split, dtype=tf.float32)

    print(f'x_train shape: {np.array(x_split).shape}')

    return np.array(x_split), np.array(org)


class LSTMClassifier(tf.keras.Model):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

        self.dense_M1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_M2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)

        self.dense_Tsk1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_Tsk2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)

        self.dense_S1 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_S2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)

        self.lstm = tf.keras.layers.LSTM(units=128, activation=tf.nn.leaky_relu, return_sequences=True)

        self.dense_PMV1 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV5 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        # 1, 25, 1 , 1, 2, 1, 1, 1, 1, 1, 1, 1,1
        # count, person, griffith_avg, season, diff, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg
        data = inputs['feature']
        body = data[0:, 0:, 33:37]
        env = data[0:, 0:, 28:33]
        Ta = data[0:, 0:, 31:32]
        Pa = tf.math.log1p(Ta)

        M_input = self.drop(body, training=training)
        M = self.dense_M1(M_input)
        M = self.drop(M, training=training)
        M = self.dense_M2(M)

        Tsk_input = self.drop(data, training=training)
        Tsk = tf.abs(self.dense_Tsk1(Tsk_input))
        Tsk_input = self.drop(Tsk, training=training)
        Tsk = tf.abs(self.dense_Tsk2(Tsk_input))
        Psk = tf.math.log1p(Tsk)

        # M, Tsk, Psk, Pa,
        # season, diff, va, ta, hr
        # age_avg, height_avg, weight_avg, bmi_avg

        s_input = []
        for (m, tsk, psk, pa, e, b) in zip(M, Tsk, Psk, Pa, env, body):
            s_input.append(tf.concat([m, tsk, psk, pa, e, b], axis=1))

        s_input = self.drop(s_input, training=training)
        S = self.dense_S1(s_input)
        s_input = self.drop(S, training=training)
        S = self.dense_S2(s_input)


        # M, Tsk, Psk, Pa, S
        # count, person, griffith_avg, season, diff, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg
        lstm_input = []
        for (m, tsk, psk, pa, s, d) in zip(M, Tsk, Psk, Pa, S, data):
            lstm_input.append(tf.concat([m, tsk, psk, pa, s, d], axis=1))

        lstm_input = self.drop(lstm_input, training=training)
        lstm = self.lstm(lstm_input)


        dense = self.dense_PMV1(lstm)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV5(dense)

        output = tf.nn.softmax(dense)[:, 2:3, :]

        # season, diff1, diff2, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg
        data = data[:, 2:3, 27:]

        x = []
        for (d, o) in zip(data, output):
            x.append(tf.concat((d, o), axis=1))
        x = tf.reshape(x, [len(data), 1, 13])

        return [output, x]


def R_loss(y_true, input):
    # season, diff1, diff2, va, ta, hr, age_avg, height_avg, weight_avg, bmi_avg, y_pred
    input = tf.squeeze(input, axis=1)
    data = scaler.inverse_transform(input[:, 3:10])
    ta = data[:, 4:5]
    # ta hr va age height weight bmi griffith gender pmv y_pred
    y_pred = input[:, 10:]
    season = input[:, 0:1]
    y_exp = []
    # ta 映射
    for i in range(0, len(ta)):
        if season[i] == 0:  # 夏季
            if 26.5 >= ta[i] >= 24:
                y_exp.append(1)
            elif ta[i] < 24:
                y_exp.append(0)
            else:
                y_exp.append(2)
        else:
            if 25.5 >= ta[i] >= 22:
                y_exp.append(1)
            elif ta[i] < 22:
                y_exp.append(0)
            else:
                y_exp.append(2)
    y_exp = tf.one_hot(y_exp, depth=3)
    total = 0
    for i in range(0, len(y_pred)):
        p_true = tf.reshape(1 - y_exp[i], [1, 3])
        p_pred = tf.reshape(tf.math.log(alpha + y_pred[i]), [3, 1])
        r = tf.matmul(p_true, p_pred)
        total += r.numpy().item()
    r_loss = beta * total / len(y_pred)
    return r_loss


def CE_loss(y_true, y_pred):
    # print(f'y_pred shape: {y_pred.shape}')
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss


def Accuracy(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=1)
    y_true = tf.reshape(y_true, [len(y_true), 1])
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_pred, y_true)


def test():
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('../../LSTM/DM/phy_lstm_loss.ckpt-1').expect_partial()
    y_pred = model({'feature': x_test}, training=False)
    y_pred = tf.squeeze(y_pred[0], axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = pd.DataFrame(data=y_pred)
    return y_pred


if __name__ == '__main__':

    scaler = MinMaxScaler()

    test_size, val_size = 0.2, 0.1

    num_epochs, batch_size, learning_rate = 128, 64, 0.008

    alpha, beta = 0.3, 1

    season = ['summer', 'winter']
    for s in season:
        filepath = s
        x_test, org = data_loader()
        model = LSTMClassifier()
        org = org.reshape([len(org), 2])
        org = pd.DataFrame(data=org)
        res = test()
        res = pd.DataFrame(data=res)
        res = pd.concat([org, res], axis=1)

        res.to_csv(f'../PLOT/dataset/{filepath}.csv', index=False)