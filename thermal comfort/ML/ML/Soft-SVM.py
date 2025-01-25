# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：Soft-SVM.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/8 22:45 
"""

# -*- coding: utf-8 -*-
import pandas as pd

from weight import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_loader():
    file_path = "../dataset/data/dataset.csv"

    df = pd.read_csv(file_path).dropna(axis=0, how='any', inplace=False)
    df.rename(columns={'thermal sensation': 'thermal_sensation'}, inplace=True)

    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0

    w1, w2, w3 = weight_griffith(df)
    w_max = min(w1, w2, w3)
    w1, w2, w3 = w1 / w_max, w2 / w_max, w3 / w_max
    w_g = [w1, w2, w3]
    print(w_g)

    w1, w2, w3 = weight_bmi(df)
    w_max = min(w1, w2, w3)
    w1, w2, w3 = w1 / w_max, w2 / w_max, w3 / w_max
    w_b = [w1, w2, w3]
    # df = df.drop(df.index[(df.thermal_sensation == 1)])
    print(w_b)

    return df, w_b, w_g


def train_test(df):
    normal = df[['age', 'height', 'weight', 'ta', 'hr', 'bmi']]
    x = scaler.fit_transform(X=normal)
    other = df[['gender', 'season', 'griffith']]
    # df.loc[(df[y_feature] == 2), y_feature] = 1
    # df.loc[(df[y_feature] == 1), y_feature] = 0
    # df.loc[(df[y_feature] == 0), y_feature] = -1
    x = np.concatenate([x, other], axis=1)
    y = df[y_feature]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def data_drive(df):
    x_train, x_test, y_train, y_test = train_test(df)
    model = SVC(kernel=kernel, decision_function_shape='ovo')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)

    for kv in accuracy.items():
        print(kv)
    for kv in precision.items():
        print(kv)
    for kv in recall.items():
        print(kv)
    for kv in f1.items():
        print(kv)


def evaluating_indicator(y_pre, y_test):
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    # 准确率
    accuracy.update({'测试集准确率：': accuracy_score(y_test, y_pre)})

    # 精确率
    precision.update({'精确率-macro：': precision_score(y_test, y_pre, average='macro')})
    precision.update({'精确率-micro：': precision_score(y_test, y_pre, average='micro')})
    precision.update({'精确率-weighted：': precision_score(y_test, y_pre, average='weighted')})
    # precision.update({'精确率-None：': precision_score(y_test, y_pre, average=None)})

    # 召回率
    recall.update({'召回率-macro：': recall_score(y_test, y_pre, average='macro')})
    recall.update({'召回率-micro：': recall_score(y_test, y_pre, average='micro')})
    recall.update({'召回率-weighted：': recall_score(y_test, y_pre, average='weighted')})
    # recall.update({'召回率-None：': recall_score(y_test, y_pre, average=None)})

    # F1 score
    f1.update({'F1 score-macro：': f1_score(y_test, y_pre, average='macro')})
    f1.update({'F1 score-micro：': f1_score(y_test, y_pre, average='micro')})
    f1.update({'F1 score-weighted：': f1_score(y_test, y_pre, average='weighted')})

    return accuracy, precision, recall, f1


def svm(data, weight):
    precision_all = []
    recall_all = []
    f1_all = []
    total = 0
    accuracy_count = 0

    for i in range(0, 3):
        df = data[i]
        c = weight[i]
        x_train, x_test, y_train, y_test = train_test(df)
        model = SVC(kernel=kernel, decision_function_shape='ovr', C=c)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy, precision, recall, f1 = evaluating_indicator(y_pred, y_test)
        total += len(x_test)
        accuracy_count += accuracy.get('测试集准确率：') * len(x_test)
        precision_all.append(list(precision.values()))
        recall_all.append(list(recall.values()))
        f1_all.append(list(f1.values()))

    print("测试集准确率为：" + str(accuracy_count / total))

    p_macro, p_micro, p_weight = 0, 0, 0
    r_macro, r_micro, r_weight = 0, 0, 0
    f1_macro, f1_micro, f1_weight = 0, 0, 0

    for i in range(0, 3):
        p_macro += 1 / 3 * precision_all[i][0]
        p_micro += 1 / 3 * precision_all[i][1]
        p_weight += 1 / 3 * precision_all[i][2]
        r_macro += 1 / 3 * recall_all[i][0]
        r_micro += 1 / 3 * recall_all[i][1]
        r_weight += 1 / 3 * recall_all[i][2]
        f1_macro += 1 / 3 * f1_all[i][0]
        f1_micro += 1 / 3 * f1_all[i][1]
        f1_weight += 1 / 3 * f1_all[i][2]
    print("精确率-macro:" + str(p_macro))
    print("精确率-micro:" + str(p_micro))
    print("精确率-weight:" + str(p_weight))
    print("召回率-macro:" + str(r_macro))
    print("召回率-micro:" + str(r_micro))
    print("召回率-weight:" + str(r_weight))
    print("F1 score-macro:" + str(f1_macro))
    print("F1 score-micro:" + str(f1_micro))
    print("F1 score-weight:" + str(f1_weight))


if __name__ == '__main__':

    x_feature = ['gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'bmi', 'griffith']
    y_feature = 'thermal_sensation'

    alpha, beta = 0.5, 0.5
    b_down, b_upper = 18.5, 25
    g_down, g_upper = 1.0, 2.0
    scaler = MinMaxScaler()
    kernel = 'linear'

    df, bmi_weight, griffith_weight = data_loader()


    bmi1, bmi2, bmi3 = split(df, 'bmi')

    data = [bmi1, bmi2, bmi3]

    print("无权重")

    data_drive(df)

    # svm(data, [1, 1, 1])

    print("\n权重为 bmi")
    svm(data, bmi_weight)

    print("\n权重系数为grffith")
    svm(data, griffith_weight)








