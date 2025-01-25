# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


def train():
    """Soft-SVM"""
    for i in range(0, 3):
        svc = SVC(kernel='rbf', decision_function_shape='ovr')
        svc.fit(x_train[i], y_train[i])
        joblib.dump(svc, f'./DM/soft_svm_{i}.pkl')

        """KNN"""
        knn = KNeighborsClassifier(
            n_neighbors=7,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=None
        )

        knn.fit(x_train[i], y_train[i])
        joblib.dump(knn, f'./DM/knn_{i}.pkl')

        """Random Forest"""
        rf = RandomForestClassifier(
            max_depth=15,
            random_state=2019
        )
        rf.fit(x_train[i], y_train[i])
        joblib.dump(rf, f'./DM/random_forest_{i}.pkl')

        """Xgboost"""
        xgb = LGBMClassifier(
            learning_rate=0.0075,
            max_depth=15,
            n_estimators=2048,
            num_leaves=63,
            random_state=2019,
            n_jobs=-1,
            reg_alpha=0.8,
            reg_lambda=0.8,
            subsample=0.2,
            colsample_bytree=0.5,
        )
        xgb.fit(x_train[i], y_train[i])
        joblib.dump(rf, f'./DM/xgboost_{i}.pkl')


def evaluate(y_pre, y_test):
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


def result(accuracy, precision, recall, f1):
    print("测试集准确率为：" + str(accuracy))

    p_macro, p_micro, p_weight = 0, 0, 0
    r_macro, r_micro, r_weight = 0, 0, 0
    f1_macro, f1_micro, f1_weight = 0, 0, 0

    for i in range(0, 3):
        p_macro += 1 / 3 * precision[i][0]
        p_micro += 1 / 3 * precision[i][1]
        p_weight += 1 / 3 * precision[i][2]
        r_macro += 1 / 3 * recall[i][0]
        r_micro += 1 / 3 * recall[i][1]
        r_weight += 1 / 3 * recall[i][2]
        f1_macro += 1 / 3 * f1[i][0]
        f1_micro += 1 / 3 * f1[i][1]
        f1_weight += 1 / 3 * f1[i][2]
    print("精确率-macro:" + str(p_macro))
    print("精确率-micro:" + str(p_micro))
    print("精确率-weight:" + str(p_weight))
    print("召回率-macro:" + str(r_macro))
    print("召回率-micro:" + str(r_micro))
    print("召回率-weight:" + str(r_weight))
    print("F1 score-macro:" + str(f1_macro))
    print("F1 score-micro:" + str(f1_micro))
    print("F1 score-weight:" + str(f1_weight))


def test(x_test, y_test):

    model = ['soft_svm', 'knn', 'random_forest', 'xgboost']

    for m in model:
        precision_all = []
        recall_all = []
        f1_all = []
        total = 0
        accuracy_count = 0
        print('**' * 10)
        print(m)
        print('**' * 10)
        for i in range(0, 3):
            total += len(x_test[i])
            svc = joblib.load(f'./DM/{m}_{i}.pkl')
            y_pred = svc.predict(x_test[i])
            accuracy, precision, recall, f1 = evaluate(y_pred, y_test[i])
            accuracy_count += accuracy.get('测试集准确率：') * len(x_test[i])
            precision_all.append(list(precision.values()))
            recall_all.append(list(recall.values()))
            f1_all.append(list(f1.values()))
        accuracy = accuracy_count / total
        result(accuracy, precision_all, recall_all, f1_all)


def data_load():
    df = pd.read_csv('../dataset/format/dataset.csv').dropna(axis=0, how='any', inplace=False)
    normalization = ['count', 'age', 'height', 'weight', 'ta', 'hr', 'bmi', 'bmi_avg']
    other = ['gender', 'season', 'griffith',
             'bmi_l', 'bmi_n', 'bmi_h',
             'grf_l', 'grf_n', 'grf_h', 'grf_avg']
    y_feature = 'tsv'

    feature = ['no', 'gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'tsv',
               'bmi', 'griffith', 'count', 'bmi_l', 'bmi_n', 'bmi_h', 'bmi_avg',
               'grf_l', 'grf_n', 'grf_h', 'grf_avg']

    if category == 2:
        # df = df.drop(df[df[y_feature] == 1].index, inplace=False)
        df.loc[(df[y_feature] != 1), y_feature] = 0

    if index == 'bmi':
        low = df.loc[(df['bmi'] <= 18.5)][feature].reset_index(drop=True)
        normal = df.loc[(df['bmi'] > 18.5) & (df['bmi'] < 25)][feature].reset_index(drop=True)
        high = df.loc[(df['bmi'] >= 25)][feature].reset_index(drop=True)
    elif index == 'griffith':
        low = df.loc[(df['griffith'] <= 1)][feature].reset_index(drop=True)
        normal = df.loc[(df['griffith'] > 1) & (df['griffith'] < 2)][feature].reset_index(drop=True)
        high = df.loc[(df['griffith'] >= 2)][feature].reset_index(drop=True)

    X_train, X_test, Y_train, Y_test = [], [], [], []
    data = [low, normal, high]
    i = 0
    for d in data:
        normal_data = d[normalization].reset_index(drop=True)
        other_data = d[other].reset_index(drop=True)
        y = d[y_feature].reset_index(drop=True)
        normal_data = scaler.fit_transform(normal_data)
        d = np.concatenate([normal_data, other_data, y[:, None]], axis=1)
        d = np.array(d)
        x_train, x_test, y_train, y_test = train_test_split(d[:, 0:-2], d[:, -1:], test_size=0.2)
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train.ravel())
        Y_test.append(y_test.ravel())
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':

    index = 'bmi'

    category = 3
    scaler = MinMaxScaler()
    x_train, x_test, y_train, y_test = data_load()
    train()
    test(x_test, y_test)


