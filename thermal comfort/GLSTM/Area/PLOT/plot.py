# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)


def split(data):
    hot_ta = data[(data[y_feature] == 2)][['ta']]
    hot_rh = data[(data[y_feature] == 2)][['hr']]
    cool_ta = data[(data[y_feature] == 0)][['ta']]
    cool_rh = data[(data[y_feature] == 0)][['hr']]
    com_ta = data[(data[y_feature] == 1)][['ta']]
    com_rh = data[(data[y_feature] == 1)][['hr']]
    return hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh


def summer():
    x1 = [23, 22.2]
    y1 = [40, 70]

    # fs_left = np.polyfit(x, y, 1)
    # print('f1 is :\n', fs_left)
    #
    # y1 = np.arange(40, 71, 1)
    # x1 = []
    # for i in np.arange(40, 71, 1):
    #     x1.append(round((i - fs_left[1]) / fs_left[0], 2))
    #
    x2 = [26, 25.2]
    y2 = [40, 70]

    # fs_right = np.polyfit(x, y, 1)
    # print('f1 is :\n', fs_right)

    # y2 = np.arange(40, 71, 1)
    # x2 = []
    # for i in np.arange(40, 71, 1):
    #     x2.append(round((i - fs_right[1]) / fs_right[0], 2))

    return x1, x2, y1, y2


def winter():
    x1 = [20.8, 20]
    y1 = [10, 25]

    # fs_left = np.polyfit(x, y, 1)
    # print('f1 is :\n', fs_left)
    #
    # y1 = np.arange(10, 26, 1)
    # x1 = []
    # for i in np.arange(10, 26, 1):
    #     x1.append(round((i - fs_left[1]) / fs_left[0], 2))

    x2 = [23.4, 22.6]
    y2 = [10, 25]

    # fs_right = np.polyfit(x, y, 1)
    # print('f1 is :\n', fs_right)
    #
    # y2 = np.arange(10, 26, 1)
    # x2 = []
    # for i in np.arange(10, 26, 1):
    #     x2.append(round((i - fs_right[1]) / fs_right[0], 2))

    return x1, x2, y1, y2


if __name__ == '__main__':
    y_feature = 'tsv'
    fontsize = 14
    df = pd.read_csv(f'./dataset/winter.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

    season = ['winter', 'summer', 'winterp', 'summerp']
    for s in season:
        df = pd.read_csv(f'./dataset/{s}.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
        if s == 'summer' or s == 'winter':
            df.rename(columns={'0': 'ta', '1': 'hr', '0.1': 'tsv'}, inplace=True)
        hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(df)

        if s == 'summer':
            x1, x2, y1, y2 = summer()
            plt.figure(figsize=(10, 8), dpi=100)

        else:
            x1, x2, y1, y2 = winter()
            plt.figure(figsize=(10, 8), dpi=100)


        axes = plt.subplot(111)
        label1 = axes.scatter(hot_ta, hot_rh, s=70, marker='x', c="red")
        label2 = axes.scatter(cool_ta, cool_rh, s=50, marker='+', c="blue")
        label3 = axes.scatter(com_ta, com_rh, s=70, marker=None, c="green")
        plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=fontsize, rotation=45)
        plt.xlabel(u"空气温度(℃)", fontsize=fontsize)
        plt.ylabel(u"相对湿度(%)", fontsize=fontsize)
        plt.grid(linestyle='--')
        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        axes.legend((label1, label2, label3), ("热不适", "冷不适", "舒适"), loc=3)



        x = [x1[0], x1[1], x2[1],  x2[0]]
        print(x)
        y = [y1[0], y1[1], y2[1],  y2[0]]
        print(y)
        if s == 'summer' or s == 'winter':
            ax.plot(x1, y1, 'gray', linestyle='--', marker='')
            ax.plot(x2, y2, 'gray', linestyle='--', marker='')
            plt.fill(x, y, color='green', alpha=0.2)
        plt.show()
