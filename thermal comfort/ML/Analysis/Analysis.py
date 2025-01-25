# -*- coding: utf-8 -*-
"""
@Project ：3033GRAD 
@File ：Analysis.py
@Author ：伍陆柒
@Desc ：
@Date ：3033/4/5 33:33 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.ticker as ticker


def split(data):
    hot_ta = data[(data[y_feature] == 2)][['ta']]
    hot_rh = data[(data[y_feature] == 2)][['hr']]
    cool_ta = data[(data[y_feature] == 0)][['ta']]
    cool_rh = data[(data[y_feature] == 0)][['hr']]
    com_ta = data[(data[y_feature] == 1)][['ta']]
    com_rh = data[(data[y_feature] == 1)][['hr']]
    return hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh


def hist(xlabel, xticker, data):
    # # 绘图风格
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)

    red = sns.color_palette("Set1")[0]
    bins = math.ceil(data.max()) - math.floor(data.min())
    weights = np.ones_like(np.array(data)) / float(len(data))
    # 绘制直方图
    sns.distplot(data,
                 bins=int(bins/xticker),
                 hist=True,
                 hist_kws={'color': 'red'},
                 kde=False,
                 kde_kws={
                     'color': 'darkorange',
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=False)
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xticker))
    ax.tick_params("x",
                   which="major",
                   length=15,
                   width=2.0,
                   rotation=45)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    if xlabel == 'BMI':
        plt.xlabel(xlabel, fontproperties='Times New Roman', size=fontsize)
    else:
        plt.xlabel(xlabel, size=fontsize)
    plt.ylabel(u"人数", size=fontsize)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)

    plt.tight_layout()
    plt.show()


def hist_3(data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)

    bins = max(math.ceil(hot_ta.max()) - math.floor(hot_ta.min()),
               math.ceil(com_ta.max()) - math.floor(com_ta.min()),
               math.ceil(cool_ta.max()) - math.floor(cool_ta.min())
               )

    hot_ta = np.array(hot_ta).flatten()
    com_ta = np.array(com_ta).flatten()
    cool_ta = np.array(cool_ta).flatten()

    data = [hot_ta, com_ta, cool_ta]
    plt.hist(x=data,  # 绘图数据
             bins=bins + 5,  # 指定直方图的条形数为30个
             edgecolor='w',  # 指定直方图的边框色
             color=['r', 'g', 'b'],  # 指定直方图的填充色
             label=['热不适', '舒适', '冷不适'],  # 为直方图呈现图例
             density=True,  # 是否将纵轴设置为密度，即频率
             alpha=0.8,  # 透明度
             rwidth=0.8,  # 直方图宽度百分比：0-3
             stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    # 显示图例
    plt.legend()
    # 显示图形
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    # plt.title(f'3033年{season}所有数据温度分布')
    plt.show()


def count(bmi, griffith):
    level1 = 0
    level2 = 0
    level3 = 0

    for b in bmi:
        if b <= b_down:
            level1 += 1
        elif b >= b_upper:
            level3 += 1
        else:
            level2 += 1
    print(f'BMI类别1包含人数{level1}')
    print(f'BMI类别2包含人数{level2}')
    print(f'BMI类别3包含人数{level3}')

    level1 = 0
    level2 = 0
    level3 = 0
    for g in griffith:
        if g <= g_down:
            level1 += 1
        elif g >= g_upper:
            level3 += 1
        else:
            level2 += 1
    print(f'热敏感度类别1包含人数{level1}')
    print(f'热敏感度类别2包含人数{level2}')
    print(f'热敏感度类别3包含人数{level3}')


def split_by_level(data):
    bmi1 = data[(data['bmi'] <= b_down)][x_feature]
    bmi2 = data[(data['bmi'] > b_down) & (data['bmi'] < b_upper)][x_feature]
    bmi3 = data[(data['bmi'] >= b_upper)][x_feature]
    griffith1 = data[(data['griffith'] <= g_down)][x_feature]
    griffith2 = data[(data['griffith'] > g_down) & (data['griffith'] < g_upper)][x_feature]
    griffith3 = data[(data['griffith'] >= g_upper)][x_feature]
    return bmi1, bmi2, bmi3, griffith1, griffith2, griffith3


def random_drop(df):
    bmi1, bmi2, bmi3, griffith1, griffith2, griffith3 = split_by_level(df)

    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(bmi1)
    print(f'bmi类别1中共计数据{bmi1.shape[0]}条数据')
    print(f'bmi类别1中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别1比例{round(hot_ta.shape[0] / bmi1.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别1中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别1比例为{round(com_ta.shape[0] / bmi1.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别1中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别1比例为{round(cool_ta.shape[0] / bmi1.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')


    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(bmi2)
    print(f'bmi类别2中共计数据{bmi2.shape[0]}条数据')
    print(f'bmi类别2中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别2比例{round(hot_ta.shape[0] / bmi2.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别2中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别2比例为{round(com_ta.shape[0] / bmi2.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别2中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别2比例为{round(cool_ta.shape[0] / bmi2.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')

    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(bmi3)
    print(f'bmi类别3中共计数据{bmi3.shape[0]}条数据')
    print(f'bmi类别3中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别3比例{round(hot_ta.shape[0] / bmi3.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别3中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别3比例为{round(com_ta.shape[0] / bmi3.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'bmi类别3中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别3比例为{round(cool_ta.shape[0] / bmi3.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')

    hist_3(bmi1)
    hist_3(bmi2)
    hist_3(bmi3)

    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(griffith1)
    print(f'griffith类别1中共计数据{griffith1.shape[0]}条数据')
    print(f'griffith类别1中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别1比例{round(hot_ta.shape[0] / griffith1.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别1中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别1比例为{round(com_ta.shape[0] / griffith1.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别1中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别1比例为{round(cool_ta.shape[0] / griffith1.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')

    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(griffith2)
    print(f'griffith类别2中共计数据{griffith2.shape[0]}条数据')
    print(f'griffith类别2中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别2比例{round(hot_ta.shape[0] / griffith2.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别2中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别2比例为{round(com_ta.shape[0] / griffith2.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别2中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别2比例为{round(cool_ta.shape[0] / griffith2.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')

    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(griffith3)
    print(f'griffith类别3中共计数据{griffith3.shape[0]}条数据')
    print(f'griffith类别3中热不适数据{hot_ta.shape[0]}条数据，'
          f'占类别3比例{round(hot_ta.shape[0] / griffith3.shape[0], 2)}'
          f'占总数据比例{round(hot_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别3中舒适数据{com_ta.shape[0]}条数据，'
          f'占类别3比例为{round(com_ta.shape[0] / griffith3.shape[0], 2)}'
          f'占总数据比例为{round(com_ta.shape[0] / df.shape[0], 2)}')
    print(f'griffith类别3中冷不适数据{cool_ta.shape[0]}条数据，'
          f'占类别3比例为{round(cool_ta.shape[0] / griffith3.shape[0], 2)}'
          f'占总数据比例为{round(cool_ta.shape[0] / df.shape[0], 2)}')


    hist_3(griffith1)
    hist_3(griffith2)
    hist_3(griffith3)


if __name__ == '__main__':
    # font()
    fontsize = 14
    df = pd.read_csv('../dataset/data/dataset.csv').dropna(axis=0, how='any', inplace=False)
    print(f'数据集总条数{df.shape[0]}')
    print(f'数据集女性数据{df.loc[df["gender"] == 0].shape[0]}')
    print(f'数据集男性数据{df.loc[df["gender"] == 1].shape[0]}')
    no = df['no'].unique()
    bmi = []
    griffith = []
    female = 0
    male = 0
    for n in no:
        data = df.loc[(df['no'] == n)]
        griffith.append((data['griffith'].unique()[0]))
        bmi.append((data['bmi'].unique()[0]))

        if data['gender'].unique()[0] == 0:
            female += 1
        else:
            male += 1
    print(f'female{female}, male{male}')

    print(griffith)
    hist(u'BMI', 0.5, np.array(bmi))
    hist(u'热敏感度', 0.2, np.array(griffith))
    b_down = 18.5
    b_upper = 25
    g_down = 1.0
    g_upper = 2.0
    count(bmi, griffith)
    x_feature = ['no', 'gender', 'age', 'height', 'weight', 'ta', 'hr', 'season', 'thermal sensation', 'bmi', 'griffith']
    y_feature = 'thermal sensation'

    random_drop(df)






