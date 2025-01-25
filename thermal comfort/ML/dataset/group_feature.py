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
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import sample
import os
import seaborn as sns


def group_feature(df):

        count = len(df['no'].unique())
        print(count)
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

        extend = [count,
                  male, female,
                  young, old,
                  short, medium, tall, height_avg,
                  thin, normal, fat, weight_avg,
                  bmi_low, bmi_normal, bmi_high, bmi_avg,
                  griffith_low, griffith_normal, griffith_high, griffith_avg]

        name = ['count',
                'male', 'female',
                'young', 'old',
                'short', 'medium', 'tall', 'height_avg',
                'thin', 'normal', 'fat', 'weight_avg',
                'bmi_l', 'bmi_n', 'bmi_h', 'bmi_avg',
                'grf_l', 'grf_n', 'grf_h', 'grf_avg']

        extend_frame = pd.DataFrame(columns=name, data=[extend]*df.shape[0])

        return extend_frame

if __name__ == '__main__':

    df = pd.read_csv('./data/Dataset.csv').dropna(axis=0, how='any', inplace=False)
    df.rename(columns={'thermal sensation': 'tsv'}, inplace=True)
    extend_frame = group_feature(df)
    print(extend_frame.columns)
    df = pd.concat([df, extend_frame], axis=1)
    print(df.columns)

    df.loc[(df['tsv'] > 0.5), 'tsv'] = 2
    df.loc[((-0.5 <= df['tsv']) & (df['tsv'] <= 0.5)), 'tsv'] = 1
    df.loc[(df['tsv'] < -0.5), 'tsv'] = 0

    print(df.columns)

    df.to_csv('./format/dataset.csv', index=False)






