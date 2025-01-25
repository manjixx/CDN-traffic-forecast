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
    uncom_ta = data[(data[y_feature] != 1)][['ta']]
    uncom_rh = data[(data[y_feature] != 1)][['hr']]
    com_ta = data[(data[y_feature] == 1)][['ta']]
    com_rh = data[(data[y_feature] == 1)][['hr']]
    return uncom_ta, uncom_rh, com_ta, com_rh


def summer():
    x1 = [23, 22.2]
    y1 = [50, 75]
    x2 = [26, 25.2]
    y2 = [50, 75]

    return x1, x2, y1, y2


if __name__ == '__main__':
    y_feature = 'tsv'
    fontsize = 28

    season = ['pmv', 'data', 'griffith', 'bmi']
    for s in season:
        df = pd.read_csv(f'./dataset/{s}.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

        dis_ta, dis_rh, com_ta, com_rh = split(df)
        plt.figure(figsize=(15, 12), dpi=80)

        axes = plt.subplot(111)
        label1 = axes.scatter(dis_ta, dis_rh, s=90, marker='x', c="orangered")
        label2 = axes.scatter(com_ta, com_rh, s=70, marker=None, c="green")
        plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=fontsize, rotation=45)
        plt.xlabel(u"空气温度(℃)", fontsize=fontsize)
        plt.ylabel(u"相对湿度(%)", fontsize=fontsize)
        plt.grid(linestyle='--')
        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        axes.legend((label1, label2), ("不适", "舒适"), loc=1, prop={"size": fontsize-2})
        x1, x2, y1, y2 = summer()
        x = [x1[0], x1[1], x2[1],  x2[0]]
        y = [y1[0], y1[1], y2[1],  y2[0]]

        ax.plot(x1, y1, 'gray', linestyle='--', marker='')
        ax.plot(x2, y2, 'gray', linestyle='--', marker='')
        plt.fill(x, y, color='green', alpha=0.2)
        plt.show()
