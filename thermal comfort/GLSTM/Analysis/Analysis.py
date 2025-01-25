# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontManager
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')




def font():
    mpl_fonts = set(f.name for f in FontManager().ttflist)
    print('all font list get from matplotlib.font_manager:')
    for f in sorted(mpl_fonts):
        print('\t' + f)


def split(data):
    hot_ta = data[(data[y_feature] == 2)][['ta']]
    hot_rh = data[(data[y_feature] == 2)][['hr']]
    cool_ta = data[(data[y_feature] == 0)][['ta']]
    cool_rh = data[(data[y_feature] == 0)][['hr']]
    com_ta = data[(data[y_feature] == 1)][['ta']]
    com_rh = data[(data[y_feature] == 1)][['hr']]
    return hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh


def distribution(title, data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    # 绘制分布图
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    label1 = axes.scatter(hot_ta, hot_rh, s=50, marker=None, c="red")
    label2 = axes.scatter(cool_ta, cool_rh, s=50, marker='x', c="blue")
    label3 = axes.scatter(com_ta, com_rh, s=50, marker='+', c="green")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)
    axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    plt.show()


def hist(season, data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    # # 绘图风格

    red = sns.color_palette("Set1")[0]
    bins = math.ceil(hot_ta.max()) - math.floor(hot_ta.min()) + 5
    # 绘制直方图
    sns.distplot(hot_ta,
                 bins=bins,
                 hist=True,
                 hist_kws={'color': red},
                 kde_kws={
                     'color': 'darkred',
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=True)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)

    # plt.title(u'2021年'+season+'热不适温度分布')
    plt.show()

    bins = math.ceil(com_ta.max()) - math.floor(com_ta.min()) + 5
    sns.distplot(com_ta,
                 bins=bins,
                 hist=True,
                 hist_kws={'color': 'green'},
                 kde_kws={
                     'color': 'darkgreen',
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=True)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)
    # plt.title(f'2021年{season}舒适温度分布')
    plt.show()

    bins = math.ceil(cool_ta.max()) - math.floor(cool_ta.min()) + 5
    sns.distplot(cool_ta,
                 bins=bins,
                 hist=True,
                 hist_kws={'color': 'dodgerblue'},
                 kde_kws={
                     'color': 'darkblue',
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=True)
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)
    plt.show()

    bins = max(math.ceil(hot_ta.max()) - math.floor(hot_ta.min()),
               math.ceil(com_ta.max()) - math.floor(com_ta.min()),
               math.ceil(cool_ta.max()) - math.floor(cool_ta.min())
              )

    hot_ta = np.array(hot_ta).flatten()
    com_ta = np.array(com_ta).flatten()
    cool_ta = np.array(cool_ta).flatten()

    data = [hot_ta, com_ta, cool_ta]
    plt.hist(x=data,  # 绘图数据
             bins=25,  # 指定直方图的条形数为20个
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
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)
    # plt.title(f'2021年{season}所有数据温度分布')
    plt.show()


def filter(title, df):
    print(title)
    no = np.array(df['no'].unique())
    len = {}
    for n in no:
        # if n <= 30:
        #     continue
        data = df.loc[df['no'] == n]
        l = data.shape[0]
        # if l <= 14:
        #     continue
        len.update({n: l})

        distribution(f'第{n}号实验人员数据分布', data)
    for key, value in len.items():
        print('%s:%s' % (key, value))


def stat(data):
    max = math.ceil(data.max())
    min = math.floor(data.min())
    bins = range(min, max)
    res = pd.cut(data['ta'], bins=bins)
    dic = res.value_counts(normalize=False, ascending=False, bins=None, dropna=True).to_dict()
    dict = {}
    sum = 0
    for k, v in dic.items():
        dict.update({k.left: v})
        sum += v
    x = []
    y = []
    for k, v in sorted(dict.items(), key=lambda x:x[0]):
        x.append(k)
        y.append(v/sum)
    return x, y


def plt_bar(x, y, color, width):
    print(x)
    print(y)

    plt.bar(x, y, width=width, color=color, align='edge', edgecolor='white')
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    plt.yticks(fontproperties='Times New Roman', size=fontsize)  # 设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=fontsize)
    plt.xlabel(u"温度(℃)", fontsize=fontsize)
    plt.ylabel(u"数据比例", fontsize=fontsize)

    plt.grid(True, axis='y', ls='dashed')

    plt.show()


def count(season, data):
    print(f'2021年{season}数据分布情况')
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    print(f'2021年{season}热不适：{len(hot_ta)}, 占比为{len(hot_ta)/len(data)}\n'
          f'舒适：{len(com_ta)}, 占比{len(com_ta)/len(data)}\n'
          f'冷不适：{len(cool_ta)}，占比{len(cool_ta)/len(data)}')

    print(f'热不适温度区间为{hot_ta.min()[0], hot_ta.max()[0]}')
    print(f'舒适温度区间为{com_ta.min()[0], com_ta.max()[0]}')
    print(f'冷不适温度区间为{round(cool_ta.min()[0], 2), cool_ta.max()[0]}')

    x1, y1 = stat(hot_ta)
    x2, y2 = stat(com_ta)
    x3, y3 = stat(cool_ta)
    max_width = max(np.diff(x1).min(), np.diff(x2).min(), np.diff(x3).min())
    print('热不适温度范围与各个温度占比为：')
    plt_bar(x1, y1, color='r', width=max_width)
    print('舒适温度范围与各个温度占比为：')
    plt_bar(x2, y2, color='g', width=max_width)
    print('冷不适温度范围与各个温度占比为：')
    plt_bar(x3, y3, color='b', width=max_width)


def feature(name, color, xlable):
    data = df.drop_duplicates(subset=['no', name]).sort_values([name])[name]
    print(np.array(data.drop_duplicates()))
    bins = data.drop_duplicates().shape[0] - 2
    avg = sum(data) / len(data)
    print('均值为' + name + f': {avg}')
    sns.distplot(data,
                 bins=bins,
                 hist=True,
                 hist_kws={'color': color[0]},
                 kde_kws={
                     'color': color[1],
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=True)
    plt.legend()
    # 显示图形
    plt.xlabel(xlable)
    plt.ylabel(u"数据比例")
    plt.show()


def plt_dis(df):
    age = df.drop_duplicates(subset=['no', 'age'])['age']
    height = df.drop_duplicates(subset=['no', 'height'])['height']
    weight = df.drop_duplicates(subset=['no', 'weight'])['weight']
    griffith = df.drop_duplicates(subset=['no', 'griffith'])['griffith']
    print(age.mean(), height.mean(), weight.mean(), griffith.mean())
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=80)
    axs = axs.ravel()
    for i in range(0, len(axs)):
        axs[i].tick_params(axis="both", labelsize=fontsize)
        axs[i].set_ylabel(u"数据比例", fontdict={'size': fontsize})

    bins = math.ceil(age.max()) - math.floor(age.min())
    axs[0].hist(x=age,  # 绘图数据
             bins=bins,  # 指定直方图的条形数为20个
             edgecolor='w',  # 指定直方图的边框色
             color=['r'],  # 指定直方图的填充色
             density=True,  # 是否将纵轴设置为密度，即频率
             alpha=0.8,  # 透明度
             rwidth=0.8,  # 直方图宽度百分比：0-1
             stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    axs[0].set_xlabel(u"年龄", fontdict={'size': fontsize})
    bins = math.ceil(height.max()) - math.floor(height.min()) - 20
    axs[1].hist(x=height,  # 绘图数据
                bins=bins,  # 指定直方图的条形数为20个
                edgecolor='w',  # 指定直方图的边框色
                color=['g'],  # 指定直方图的填充色
                density=True,  # 是否将纵轴设置为密度，即频率
                alpha=0.8,  # 透明度
                rwidth=0.8,  # 直方图宽度百分比：0-1
                stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    axs[1].set_xlabel(u"身高", fontdict={'size': fontsize})
    bins = math.ceil(weight.max()) - math.floor(weight.min()) - 35
    axs[2].hist(x=weight,  # 绘图数据
                bins=bins,  # 指定直方图的条形数为20个
                edgecolor='w',  # 指定直方图的边框色
                color=['b'],  # 指定直方图的填充色
                density=True,  # 是否将纵轴设置为密度，即频率
                alpha=0.8,  # 透明度
                rwidth=0.8,  # 直方图宽度百分比：0-1
                stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    axs[2].set_xlabel(u"体重", fontdict={'size': fontsize})
    bins = math.ceil(griffith.max()) - math.floor(griffith.min())
    axs[3].hist(x=griffith,  # 绘图数据
                bins=bins,  # 指定直方图的条形数为20个
                edgecolor='w',  # 指定直方图的边框色
                color=['m'],  # 指定直方图的填充色
                density=True,  # 是否将纵轴设置为密度，即频率
                alpha=0.8,  # 透明度
                rwidth=0.8,  # 直方图宽度百分比：0-1
                stacked=True)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放
    axs[3].set_xlabel(u"客观热敏感度", fontdict={'size': fontsize})
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    fontsize = 12
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)
    # font()
    # 未经过序列化数据
    df = pd.read_csv('../../DataSet/synthetic.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

    print(f'数据总量{df.shape[0]}')
    y_feature = 'thermal sensation'
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0
    plt_dis(df)
    # winter = df.loc[(df['season'] == 'winter')].reset_index(drop=True)
    # summer = df.loc[(df['season'] == 'summer')].reset_index(drop=True)
    # print(f'冬季数据数量{winter.shape[0]}')
    # print(f'夏季数据数量{summer.shape[0]}')
    #
    # print('\n-------------------------------------\n')
    # # distribution('2021夏季数据分布图', summer)
    # hist('夏季', summer)
    # # count('夏季', summer)
    # # print('\n-------------------------------------\n')
    #
    # # distribution('2021冬季数据分布图', winter)
    # hist('冬季', winter)
    # # count('冬季', winter)
    # print('\n-------------------------------------\n')
    # # color = ['b', 'darkblue']
    # # feature('age', color, '年龄')
    # print('\n-------------------------------------\n')


