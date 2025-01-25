# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(data):
    # data 数据类型为dataframe
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False
    # 绘图风格
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)
    # 调色板取颜色
    red = sns.color_palette("Set1")[0]
    # 直方图柱子条数
    bins = math.ceil(data.max()) - math.floor(data.min()) + 5
    # 绘制直方图
    sns.distplot(data,
                 bins=bins,
                 hist=True,
                 hist_kws={'color': red},   # 柱子颜色 [green, darkgreen], [blue, darkblue]
                 kde_kws={
                     'color': 'darkred',    # 概率曲线颜色
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=True)
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    # plt.title(u'2021年'+season+'热不适温度分布')
    plt.show()


def plot_hist(data1, data2, data3):
    # data 数据类型为dataframe
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False

    bins = max(math.ceil(data1.max()) - math.floor(data1.min()),
               math.ceil(data2.max()) - math.floor(data2.min()),
               math.ceil(data3.max()) - math.floor(data3.min()))

    data1 = np.array(data1).flatten()
    data2 = np.array(data2).flatten()
    data3 = np.array(data3).flatten()

    data = [data1, data2, data3]
    plt.hist(x=data,  # 绘图数据
             bins=bins + 5, # 指定直方图的条形数为20个
             edgecolor='w',  # 指定直方图的边框色
             color=['r', 'g', 'b'],  # 指定直方图的填充色
             label=['热不适', '舒适', '冷不适'],  # 为直方图呈现图例
             density=True,  # 是否将纵轴设置为密度，即频率
             alpha=0.8,  # 透明度
             rwidth=0.8,  # 直方图宽度百分比：0-1
             stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    # 显示图例
    plt.legend()
    # 显示图形
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    # plt.title(f'2021年{season}所有数据温度分布')
    plt.show()

