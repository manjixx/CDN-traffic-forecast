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
    env = np.load('../dataset/synthetic/env.npy', allow_pickle=True).astype(float)  # ta hr va
    print(len(env))
    season = np.load('../dataset/synthetic/season.npy', allow_pickle=True).astype(int)  # season
    date = np.load('../dataset/synthetic/date.npy', allow_pickle=True)  # date
    body = np.load('../dataset/synthetic/body.npy', allow_pickle=True).astype(float)  # age height weight bmi
    # griffith, gender, sensitivity, preference, environment
    gender = np.load('../dataset/synthetic/gender.npy', allow_pickle=True)
    gender = gender[:, 0:2]
    label = np.load('../dataset/synthetic/label.npy', allow_pickle=True).astype(int)  # pmv

    # normalization: [ta hr va age height weight bmi]
    x = np.concatenate((env, body), axis=1)
    x = scaler.fit_transform(x)
    # season ta hr va age height weight bmi griffith, gender pmv
    x = np.concatenate((season, x, gender, label), axis=1)

    x_split = []
    y_split = []
    for i in range(0, round(len(x) / 7)):
        start = i * 7
        end = (i + 1) * 7
        x_hat = x[start: end, :]
        y_hat = label[start: end, :]
        for j in range(0, 4):
            x_split.append(x_hat[j: j + 3, :])
            y_split.append(y_hat[j + 2: j + 3, :])

    x_train, x_test, y_train, y_test = train_test_split(x_split, y_split, test_size=test_size)
    print(f'x_train shape: {np.array(x_train).shape}')
    print(f'y_train shape: {np.array(y_train).shape}')
    print(f'x_test shape: {np.array(x_test).shape}')
    print(f'y_test shape: {np.array(y_test).shape}')

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


class Classifier_Modeling(tf.keras.Model):
    def __init__(self):
        super(Classifier_Modeling, self).__init__()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

        self.dense_PMV1 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)
        self.dense_PMV5 = tf.keras.layers.LSTM(units=128, activation=tf.nn.leaky_relu, return_sequences=True)
        self.dense_PMV6 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)
        self.dense_PMV7 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_PMV8 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV9 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV10 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)


    def call(self, inputs, training=None, mask=None):
        # [season ta hr va age height weight bmi griffith, gender, sensitivity, preference, environment]
        data = inputs['feature']
        dense = self.drop(data, training=training)
        dense = self.dense_PMV1(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV5(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV6(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV7(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV8(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV9(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV10(dense)

        output = tf.nn.softmax(dense)[:, 2:3, :]
        return output


def CE_loss(y_true, y_pred):
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss


def Accuracy(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=1)
    y_true = tf.reshape(y_true, [len(y_true), 1])
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_pred, y_true)


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [Accuracy]
    loss = [CE_loss]
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='CE_loss', min_delta=0.0001, patience=100, verbose=1,
                                                 mode='min', restore_best_weights=True)
    callbacks = [earlyStop]
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x={'feature': x_train},
              y=y_train,
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
    checkpoint.restore('DM/model_lstm.ckpt-1').expect_partial()
    y_pred = model({'feature': x_test}, training=False)
    y_pred = tf.squeeze(y_pred[0], axis=1)
    print(f'y_pred shape:{np.array(y_pred).shape}')
    print(f'y_test shape:{np.array(y_test).shape}')

    y = tf.squeeze(y_test, axis=1)
    print(f'y_test shape:{np.array(y).shape}')
    y_pred = np.argmax(y_pred, axis=1)
    print('准确率：' + str(accuracy_score(y_pred, y)))
    print('精确率 macro：' + str(precision_score(y_pred, y, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, y, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, y, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, y, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, y, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, y, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, y, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, y, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, y, average='weighted')))


if __name__ == '__main__':
    seed_tensorflow(2022)
    scaler = MinMaxScaler()

    test_size, val_size = 0.2, 0.1

    x_train, y_train, x_test, y_test = data_loader()
    num_epochs, batch_size, learning_rate = 128, 64, 0.008

    model = Classifier_Modeling()
    train()
    test()
