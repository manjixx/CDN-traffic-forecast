# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontManager
import matplotlib.ticker as ticker


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
    plt.xlabel("temp(℃)")
    plt.ylabel("humid(%)")
    axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    plt.show()


def hist(season, data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    # # 绘图风格
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)

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
    plt.xlabel(u"温度(℃)")
    # max = math.ceil(hot_ta.max())
    # min = math.floor(hot_ta.min())
    # x = np.arange(min, max, 1)
    # plt.xticks(x, rotation=45)
    plt.ylabel(u"数据比例")
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
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
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
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    # plt.title(f'2021年{season}冷不适温度分布')
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
             bins=bins,  # 指定直方图的条形数为20个
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

def load(year):
    if year == '2018':    # 数据清洗
        df = pd.read_csv('../../dataset/2018.csv').dropna(axis=0, how='any', inplace=False)
        data = df[(df.time != '8:50:00') & (df.time != '14:20:00') & (df.time != '18:00:00')]
        data = data.drop(data.index[(data.no == 3) & (data.date == '2018/7/16')])
        data = data.drop(data.index[(data.no == 6) & (data.date == '2018/7/16')])
        data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
        data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'
    elif year == '2019_summer':
        df = pd.read_csv('../../dataset/2019_summer_clean.csv').dropna(axis=0, how='any', inplace=False)
        data = df.drop(df.index[(df.time == '12:30:00') | (df.time == '13:00:00') |
                                (df.time == '13:30:00') | (df.time == '14:00:00') |
                                (df.time == '18:00:00')])
        data = data.drop(data.index[(data.no == 9) & (data.date == '2019/7/29')])
        data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
        data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'
        data = data.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False)
    elif year == '2019_winter':
        df = pd.read_csv('../../dataset/2019_winter.csv').dropna(axis=0, how='any', inplace=False)
        data = df.drop(df.index[(df.time == '8:40:00') | (df.time == '8:50:00') | (df.time == '14:10:00')])
        data.loc[(data['time'] == '9:00:00'), 'time'] = '09:00:00'
        data.loc[(data['time'] == '9:30:00'), 'time'] = '09:30:00'
        data.loc[(data['date'] == '2019/1/9'), 'date'] = '2019/1/09'
        data = data.sort_values(by=['no', 'date', 'time'], axis=0, ascending=True, inplace=False)
    else:
        df = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
        df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
        df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
        data = df.drop(df.index[(df.time > '17:30:00') | (df.time >= '12:30:00') & (df.time <= '14:00:00')])
        data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 20) & (data.room == 1)])
        data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 56) & (data.room == 1)])
        data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 25) & (data.room == 1)])
        data = data.drop(data.index[(data.date == '2021/7/29') & (data.no == 33) & (data.room == 1)])
        data = data.drop(data.index[(data.date == '2021/7/25') & (data.no == 49) & (data.time == '12:00:00')])

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    return data


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


if __name__ == '__main__':
    # font()
    # 未经过序列化数据
    df = pd.read_csv('../../DataSet/synthetic.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    # df = pd.read_csv('../../ODataSet/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)

    print(f'数据总量{df.shape[0]}')
    y_feature = 'thermal sensation'
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0

    winter = df.loc[(df['season'] == 'winter')].reset_index(drop=True)
    summer = df.loc[(df['season'] == 'summer')].reset_index(drop=True)
    print(f'冬季数据数量{winter.shape[0]}')
    print(f'夏季数据数量{summer.shape[0]}')
    # distribution('2021夏季数据分布图', summer)
    distribution('2021冬季数据分布图', winter)
    # distribution('2021数据分布图', df)
    # hist('夏季', summer)
    hist('冬季', winter)
    # count('夏季', summer)
    count('冬季', winter)


