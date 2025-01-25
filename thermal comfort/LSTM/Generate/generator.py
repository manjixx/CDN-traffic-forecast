# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import random
import os
from random import shuffle


def load():
    df = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    df.loc[(df['date'] == '2021/1/9'), 'date'] = '2021/1/09'

    data = df.drop(df.index[(df.time > '17:30:00') | (df.time >= '12:30:00') & (df.time <= '14:00:00')])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 20) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 56) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 25) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/29') & (data.no == 33) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/25') & (data.no == 49) & (data.time == '12:00:00')])
    return data


def write_header(filepath, fieldnames):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        fieldnames = fieldnames
        writer = csv.DictWriter(fs, fieldnames=fieldnames)
        writer.writeheader()


def pmv_model(M, clo, tr, ta, vel, rh):
    """
    pmv模型接口函数
    :param M: 人体代谢率，默认为静坐状态1.1
    :param clo: 衣服隔热系数，夏季：0.5 冬季 0.8
    :param tr:
    :param ta: 室内温度
    :param vel: 风速
    :param rh: 相对湿度
    :return: pmv:计算所得pmv值
    """
    Icl = 0.155 * clo
    tcl, hc = iteration(M=M, Icl=Icl, tcl_guess=ta, tr=tr, ta=ta, vel=vel)
    if Icl <= 0.078:
        fcl = 1 + 1.29 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235))
    p1 = (0.303 * np.exp(-0.036 * M)) + 0.028
    p2 = 3.05 * 10 ** (-3) * (5733 - pa - 6.99 * M)
    p3 = 0.42 * (M - 58.15)
    p4 = 1.7 * 10 ** (-5) * M * (5.867 - pa)
    p5 = 0.0014 * M * (34 - ta)
    p_extra = (tcl + 273) ** 4 - (tr + 273) ** 4
    p6 = 3.96 * 10 ** (-8) * fcl * p_extra
    p7 = fcl * hc * (tcl - ta)

    PMV = p1 * (M - p2 - p3 - p4 - p5 - p6 - p7)

    PDD = 100 - 95 * np.exp(-0.03353 * PMV ** 4 - 0.2179 * PMV ** 2)
    # print('PDD: ' + str(PDD))
    return PMV


def iteration(M, Icl, tcl_guess, tr, ta, vel):
    if Icl <= 0.078:
        fcl = 1 + 1.29 * Icl
    else:
        fcl = 1.05 + 0.645 * Icl
    N = 0
    while True:
        N += 1
        h1 = 2.38 * (abs(tcl_guess - ta) ** 0.25)
        h2 = 12.1 * np.sqrt(vel)
        if h1 > h2:
            hc = h1
        else:
            hc = h2

        para1 = ((tcl_guess + 273) ** 4 - (tr + 273) ** 4)
        para2 = hc * (tcl_guess - ta)
        tcl_cal = 35.7 - 0.028 * M - Icl * fcl * (3.96 * 10 ** (-8) * para1 + para2)

        if abs(tcl_cal - tcl_guess) > 0.00015:
            tcl_guess = 0.5 * (tcl_guess + tcl_cal)
        else:
            break

        if N > 200:
            break
    # print(N)
    # print(tcl_cal - tcl_guess)
    # print(tcl_cal)
    return tcl_cal, hc


if __name__ == '__main__':
    filepath = '../../dataset/synthetic.csv'

    if os.access(filepath, os.F_OK):
        os.remove(filepath)

    feature = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
               'sensitivity', 'preference', 'environment',
               'thermal sensation',
               'season', 'date', 'time', 'room', 'ta', 'hr']

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'sensitivity', 'preference', 'environment',
                  'thermal sensation',
                  'season', 'date', 'time', 'room', 'ta', 'hr', 'va']
    df = load()
    winter = df.loc[(df['season'] == 'winter')].reset_index(drop=True)
    summer = df.loc[(df['season'] == 'summer')].reset_index(drop=True)
    # date = np.array(winter['date'].sort_values().unique())
    # print(date)

    wd = ['2021/1/09', '2021/1/10', '2021/1/13', '2021/1/15', '2021/1/16', '2021/1/17', '2021/1/18']

    sd = ['2021/7/20', '2021/7/21', '2021/7/22', '2021/7/23', '2021/7/24', '2021/7/25',
          '2021/7/26', '2021/7/27', '2021/7/29', '2021/7/30', '2021/7/31', '2021/8/1']

    time = ['09:00:00', '09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00',
            '14:30:00', '15:00:00', '15:30:00', '16:00:00', '16:30:00', '17:00:00', '17:30:00']

    winter_env = {}
    total_ta = 0
    num = 0
    for d in wd:
        ta_rh = []
        for t in time:
            temp = winter.loc[(winter['date'] == d) & (winter['time'] == t)][['ta', 'hr']]
            temp = temp.values.tolist()[0]
            total_ta += temp[0]
            num += 1
            ta_rh.append(temp)
        winter_env.update({d: ta_rh})
    print(winter_env)
    print(f'冬季温度平均值为{total_ta / num}')

    summer_env = {}
    total_ta = 0
    num = 0
    for d in sd:
        ta_rh = []
        for t in time:
            temp = summer.loc[(summer['date'] == d) & (summer['time'] == t)][['ta', 'hr']]
            temp = temp.values.tolist()[0]
            total_ta += temp[0]
            num += 1
            ta_rh.append(temp)
        summer_env.update({d: ta_rh})
    print(f'夏季温度平均值为{total_ta / num}')
    print(summer_env)


    no = np.array(df['no'].sort_values().unique())
    m = 1.2
    clo_w = 0.95
    clo_s = 0.5
    vel = 0.8 * round(random.random(), 1)
    vel_w = 1.5 * round(random.random(), 1)
    final = pd.DataFrame(columns=feature)
    for n in no:
        all_data = df.loc[df['no'] == n]
        winter_data = winter.loc[winter['no'] == n][feature]
        date = np.array(winter_data['date'].unique())
        create_data = []
        create_date = []
        shuffle(wd)
        if len(date) < 3:
            for d in wd:
                if d not in date:
                    create_date.append(d)
                    if len(create_date) + len(date) >= 3:
                        break
            create_date.sort()

            for cd in create_date:
                env = winter_env.get(cd)
                for i in range(0, len(env)):
                    # + round((2 - (-2)) * np.random.random() + (-2), 1)
                    ta = env[i][0] + round(random.uniform(-5.7, 2.7), 2)
                    rh = env[i][1]
                    pmv = pmv_model(M=m * 58.15, clo=clo_w, tr=ta + 0.1, ta=ta, vel=vel_w, rh=rh)
                    temp_data = all_data.iloc[0: 1, 0:10].values.flatten().tolist()
                    temp_data.append(round(pmv, 1))
                    temp_data.append('winter')
                    temp_data.append(cd)
                    temp_data.append(time[i])
                    room = np.array(all_data.iloc[0: 1, 16:17]).flatten()[0]
                    temp_data.append(room)
                    temp_data.append(ta)
                    temp_data.append(rh)
                    create_data.append(temp_data)
            create_data = pd.DataFrame(create_data, columns=feature)
            winter_data = pd.concat([winter_data.reset_index(drop=True), create_data], axis=0)
            # winter_data = pd.concat([winter_data.reset_index(drop=True), va], axis=1)
            winter_data = winter_data.sort_values(by=['date', 'time'], axis=0, ascending=True, inplace=False)
        final = final.append(winter_data, ignore_index=True)

        summer_data = summer.loc[summer['no'] == n][feature]
        summer_len = summer_data.shape[0]
        date = np.array(summer_data['date'].unique())
        create_data = []
        create_date = []
        shuffle(sd)
        if len(date) < 3:
            for d in sd:
                if d not in date:
                    create_date.append(d)
                    if len(create_date) + len(date) >= 3:
                        break
            create_date.sort()
            print(create_date)

            for cd in create_date:
                env = summer_env.get(cd)
                for i in range(0, len(env)):
                    ta = env[i][0] + round(random.uniform(-5, 2.3), 2)
                    rh = env[i][1]
                    pmv = pmv_model(M=m * 58.15, clo=clo_s, tr=ta + 0.1, ta=ta, vel=vel, rh=rh)
                    temp_data = all_data.iloc[0: 1, 0:10].values.flatten().tolist()
                    temp_data.append(round(pmv, 1))
                    temp_data.append('summer')
                    temp_data.append(cd)
                    temp_data.append(time[i])
                    room = np.array(all_data.iloc[0: 1, 16:17]).flatten()[0]
                    temp_data.append(room)
                    temp_data.append(ta)
                    temp_data.append(rh)
                    create_data.append(temp_data)
            create_data = pd.DataFrame(create_data, columns=feature)
            summer_data = pd.concat([summer_data.reset_index(drop=True), create_data], axis=0)
            # summer_data = pd.concat([summer_data.reset_index(drop=True), va], axis=1)
            summer_data = summer_data.sort_values(by=['date', 'time'], axis=0, ascending=True, inplace=False)
        final = final.append(summer_data, ignore_index=True)

    va = []
    for i in range(0, final.shape[0]):
        va.append(1.2 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    final = pd.concat([final, va], axis=1)
    final.to_csv(filepath, index=True)
    print(final)
