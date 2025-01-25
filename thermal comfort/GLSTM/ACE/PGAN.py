# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler


def seed_tensorflow(seed=2022):
    tf.get_logger().setLevel('ERROR')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def data_loader():
    env = np.load('../Dataset/synthetic/env.npy', allow_pickle=True).astype(float)  # ta hr va
    print(len(env))
    season = np.load('../Dataset/synthetic/season.npy', allow_pickle=True).astype(int)  # season
    date = np.load('../Dataset/synthetic/date.npy', allow_pickle=True)  # date
    body = np.load('../Dataset/synthetic/body.npy', allow_pickle=True).astype(float)  # age height weight bmi
    # griffith, gender, sensitivity, preference, environment
    gender = np.load('../Dataset/synthetic/gender.npy', allow_pickle=True)
    gender = gender[:, 0:2]
    y = np.load('../Dataset/synthetic/label.npy', allow_pickle=True).astype(int)  # pmv

    # normalization: [ta hr va age height weight bmi griffith]
    x = np.concatenate((env, body), axis=1)
    x = scaler.fit_transform(x)
    # season ta hr va age height weight bmi griffith gender pmv
    x = np.concatenate((season, x, gender), axis=1)

    # split train test
    train_feature, test_feature, train_label, test_label = train_test_split(x, y, test_size=0.2)
    print(f'train_feature shape: {len(train_feature)} * {len(train_feature[0])}')
    print(f'test_feature shape: {len(test_feature)} * {len(test_feature[0])}')

    return np.array(train_feature), np.array(test_feature), np.array(train_label), np.array(
        test_label)


class Classifier_Modeling(tf.keras.Model):
    def __init__(self):
        super(Classifier_Modeling, self).__init__()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

        self.dense_M1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_M2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense_Tsk1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_Tsk2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense_S1 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_S2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)

        self.dense_PMV1 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=4, activation=tf.nn.leaky_relu)
        self.dense_PMV5 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        data = inputs['feature']  # [ta, hr, va, age, weight, height, bmi, gender]
        body = data[:, 4:]  # [gender, age, weight, height, bmi]
        environment = data[:, 1:4]  # [ta, hr, va]
        T = data[:, 1:2]  # Ta
        Pa = tf.math.log1p(T)

        M_input = self.drop(body, training=training)
        M = self.dense_M1(M_input)
        M = self.drop(M, training=training)
        M = self.dense_M2(M)

        Tsk_input = self.drop(data, training=training)
        Tsk = tf.abs(self.dense_Tsk1(Tsk_input))
        Tsk_input = self.drop(Tsk, training=training)
        Tsk = tf.abs(self.dense_Tsk2(Tsk_input))

        Psk = tf.math.log1p(Tsk)

        s_input = tf.concat([body, M, Tsk, Psk, environment, Pa], axis=1)
        s_input = self.drop(s_input, training=training)
        S = self.dense_S1(s_input)
        s_input = self.drop(S, training=training)
        S = self.dense_S2(s_input)

        pmv_input = tf.concat([body, M, Tsk, Psk, environment, Pa, S], axis=1)
        dense = self.dense_PMV1(pmv_input)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV5(dense)

        output = tf.nn.softmax(dense)
        x = tf.concat((data, output), axis=1)
        return [output, x]

    def get_embedding(self):
        return self.embedding.get_weights()


def R_loss(y_true, input):
    # season ta hr va age height weight bmi griffith, gender, y_pred
    data = scaler.inverse_transform(input[:, 1:8])
    ta = data[:, 1:2]
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
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss


def Accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_pred, y_true)


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [Accuracy]
    loss = [CE_loss, R_loss]
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1,
                                                 mode='min', restore_best_weights=True)
    callbacks = [earlyStop]
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x={'feature': train_feature},
              y=[train_label, train_label],
              epochs=num_epochs,
              batch_size=batch_size,
              validation_split=0.05,
              callbacks=callbacks,
              verbose=1,
              shuffle=True)
    checkpoint = tf.train.Checkpoint(classifier=model)
    path = checkpoint.save('DM/model_ann.ckpt')
    print("model saved to %s" % path)


def test():
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('DM/model_ann.ckpt-1').expect_partial()
    y_pred = model({'feature': test_feature}, training=False)
    y_pred = np.argmax(y_pred[0], axis=1)
    print('准确率：' + str(accuracy_score(y_pred, test_label)))
    print('精确率 macro：' + str(precision_score(y_pred, test_label, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, test_label, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, test_label, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, test_label, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, test_label, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, test_label, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, test_label, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, test_label, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, test_label, average='weighted')))


if __name__ == '__main__':
    seed_tensorflow(2022)
    scaler = MinMaxScaler()

    train_feature, test_feature, train_label, test_label = data_loader()
    model = Classifier_Modeling()
    alpha, beta = 0.3, 1
    num_epochs, batch_size, learning_rate = 128, 32, 0.008

    train()
    test()
