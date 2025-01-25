# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：csv2npy_Ver1.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/5 20:18
"""

import pandas as pd
import numpy as np
import random
import os

def tocsv(df, count):

    res_list = []
    # gender
    gender = df.drop_duplicates(subset=['no', 'gender'])
    male = gender.loc[(gender['gender'] == 1)].shape[0]
    female = gender.loc[(gender['gender'] == 0)].shape[0]
    gender_max = max(male, female)
    male, female = male / gender_max, female / gender_max
    # age
    age = df.drop_duplicates(subset=['no', 'age'])
    young = age.loc[(age['age'] < 25)].shape[0]
    old = age.loc[(age['age'] >= 25)].shape[0]
    age_max = max(young, old)
    young, old = young / age_max, old / age_max
    age_avg = sum(age['age'].tolist()) / count

    # height
    height = df.drop_duplicates(subset=['no', 'height'])
    short = height.loc[(height['height'] <= 170)].shape[0]
    medium = height.loc[(height['height'] > 170) & (height['height'] < 180)].shape[0]
    tall = height.loc[(height['height'] >= 180)].shape[0]
    height_max = max(short, medium, tall)
    short, medium, tall = short / height_max, medium / height_max, tall / height_max
    height_avg = sum(height['height'].tolist()) / count

    # weight
    weight = df.drop_duplicates(subset=['no', 'weight'])
    thin = weight.loc[(weight['weight'] <= 60)].shape[0]
    normal = weight.loc[(weight['weight'] > 60) & (weight['weight'] < 75)].shape[0]
    fat = weight.loc[(weight['weight'] >= 75)].shape[0]
    weight_max = max(thin, normal, fat)
    thin, normal, fat = thin / weight_max, normal / weight_max, fat / weight_max
    weight_avg = sum(weight['weight'].tolist()) / count

    # bmi
    bmi = df.drop_duplicates(subset=['no', 'bmi'])
    bmi_low = bmi.loc[(bmi['bmi'] <= 18.5)].shape[0]
    bmi_normal = bmi.loc[(bmi['bmi'] > 18.5) & (bmi['bmi'] < 25)].shape[0]
    bmi_high = bmi.loc[(bmi['bmi'] > 25)].shape[0]
    bmi_max = max(bmi_low, bmi_normal, bmi_high)
    bmi_low, bmi_normal, bmi_high = bmi_low / bmi_max, bmi_normal / bmi_max, bmi_high / bmi_max
    bmi_avg = sum(bmi['bmi'].tolist()) / count

    # griffith

    griffith = df.drop_duplicates(subset=['no', 'griffith'])
    griffith_low = griffith.loc[(griffith['griffith'] <= 1)].shape[0]
    griffith_normal = griffith.loc[(griffith['griffith'] > 1) & (griffith['griffith'] < 2)].shape[0]
    griffith_high = griffith.loc[(griffith['griffith'] >= 2)].shape[0]
    griffith_max = max(griffith_low, griffith_normal, griffith_high)
    griffith_low, griffith_normal, griffith_high = \
        griffith_low / griffith_max, griffith_normal / griffith_max, griffith_high / griffith_max
    griffith_avg = sum(griffith['griffith'].tolist()) / count

    # sensitivity
    sensitivity = df.drop_duplicates(subset=['no', 'sensitivity'])
    sensitivity_low = sensitivity.loc[(sensitivity['sensitivity'] == 0)].shape[0]
    sensitivity_normal = sensitivity.loc[(sensitivity['sensitivity'] == 1)].shape[0]
    sensitivity_high = sensitivity.loc[(sensitivity['sensitivity'] == 2)].shape[0]
    sensitivity_max = max(sensitivity_low, sensitivity_normal, sensitivity_high)
    sensitivity_low, sensitivity_normal, sensitivity_high = \
        sensitivity_low / sensitivity_max, sensitivity_normal / sensitivity_max, sensitivity_high / sensitivity_max

    # preference
    preference = df.drop_duplicates(subset=['no', 'preference'])
    preference_low = preference.loc[(preference['preference'] == -1)].shape[0]
    preference_normal = preference.loc[(preference['preference'] == 0)].shape[0]
    preference_high = preference.loc[(preference['preference'] == 1)].shape[0]
    preference_max = max(preference_low, preference_normal, preference_high)
    preference_low, preference_normal, preference_high = \
        preference_low / preference_max, preference_normal / preference_max, preference_high / preference_max

    # environment
    environment = df.drop_duplicates(subset=['no', 'environment'])
    environment_low = environment.loc[(environment['environment'] == -1)].shape[0]
    environment_normal = environment.loc[(environment['environment'] == 0)].shape[0]
    environment_high = environment.loc[(environment['environment'] == 1)].shape[0]
    environment_max = max(environment_low, environment_normal, environment_high)
    environment_low, environment_normal, environment_high = \
        environment_low / environment_max, environment_normal / environment_max, environment_high / environment_max

    date = np.array(df['date'].unique())

    for d in date:
        data = df.loc[(df['date'] == d)]
        time = np.array(data['time'].unique())
        ta_1, ta_2, k = 0, 0, 0
        for t in time:
            dt = data.loc[(data['time'] == t)]
            ta = sum(dt['ta'].tolist()) / count
            hr = sum(dt['hr'].tolist()) / count
            if k == 0:
                diff1, diff2 = 0, 0
                ta_1 = ta
                ta_2 = 0
            elif k == 1:
                diff1 = ta - ta_1
                diff2 = 0
                ta_2 = ta_1
                ta_1 = ta
            else:
                diff1 = ta - ta_1
                diff2 = ta - ta_2
                ta_2 = ta_1
                ta_1 = ta
            k += 1
            season = dt['season'].unique()[0]
            va = 1.2 * random.uniform(0, 1)
            res = [count, male, female, young, old, short, medium, tall, thin, normal, fat,
                   bmi_low, bmi_normal, bmi_high,
                   griffith_low, griffith_normal, griffith_high,
                   sensitivity_low, sensitivity_normal, sensitivity_high,
                   preference_low, preference_normal, preference_high,
                   environment_low, environment_normal, environment_high,
                   d, t, ta, hr, season, va, diff1, diff2,
                   age_avg, height_avg, weight_avg, bmi_avg, griffith_avg,
                   ]
            res_list.append(res)
    name = ['count',
            'male', 'female',
            'young', 'old',
            'short', 'medium', 'tall',
            'thin', 'normal', 'fat',
            'bmi_l', 'bmi_n', 'bmi_h',
            'grf_l', 'grf_n', 'grf_h',
            'sen_l', 'sen_n', 'sen_h',
            'pre_l', 'pre_n', 'pre_h',
            'env_l', 'env_n', 'env_h',
            'date', 'time', 'ta', 'hr', 'season', 'va',
            'ta_diff1', 'ta_diff2',
            'age_avg', 'height_avg', 'weight_avg', 'bmi_avg', 'griffith_avg']
    resDf = pd.DataFrame(columns=name, data=res_list)
    return resDf


if __name__ == '__main__':
    df = pd.read_csv('../Synthetic/lstm.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    df.sort_values(['no', 'date', 'time'])
    season = ['summer', 'winter']
    for s in season:
        if s == 'summer':
            season = 0
        else:
            season = 1
        data = df.loc[(df['season'] == season)].reset_index(drop=True)

        lstm = tocsv(data, 10)
        lstm.to_csv(s + '.csv', index=False)

        # print(lstm






