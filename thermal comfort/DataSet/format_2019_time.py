# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：format_2019_time.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/17 23:01 
"""

import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./2019_summer.csv').dropna(axis=0, how='any', inplace=False)
    df = df.sort_values(by=['no', 'date', 'time'])
    df = df.drop_duplicates(subset=['no', 'date', 'time'], keep='first', inplace=False)

    df.loc[(df['time'] <= '09:15:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] > '09:15:00') & (df['time'] <= '09:45:00'), 'time'] = '09:30:00'
    df.loc[(df['time'] > '09:45:00') & (df['time'] <= '10:15:00'), 'time'] = '10:00:00'
    df.loc[(df['time'] > '10:15:00') & (df['time'] <= '10:45:00'), 'time'] = '10:30:00'
    df.loc[(df['time'] > '10:45:00') & (df['time'] <= '11:15:00'), 'time'] = '11:00:00'
    df.loc[(df['time'] > '11:15:00') & (df['time'] <= '11:45:00'), 'time'] = '11:30:00'
    df.loc[(df['time'] > '11:45:00') & (df['time'] <= '12:15:00'), 'time'] = '12:00:00'
    df.loc[(df['time'] > '12:15:00') & (df['time'] <= '12:45:00'), 'time'] = '12:30:00'
    df.loc[(df['time'] > '12:45:00') & (df['time'] <= '13:15:00'), 'time'] = '13:00:00'
    df.loc[(df['time'] > '13:15:00') & (df['time'] <= '13:45:00'), 'time'] = '13:30:00'
    df.loc[(df['time'] > '13:45:00') & (df['time'] <= '14:15:00'), 'time'] = '14:00:00'
    df.loc[(df['time'] > '14:15:00') & (df['time'] <= '14:45:00'), 'time'] = '14:30:00'
    df.loc[(df['time'] > '14:45:00') & (df['time'] <= '15:15:00'), 'time'] = '15:00:00'
    df.loc[(df['time'] > '15:15:00') & (df['time'] <= '15:45:00'), 'time'] = '15:30:00'
    df.loc[(df['time'] > '15:45:00') & (df['time'] <= '16:15:00'), 'time'] = '16:00:00'
    df.loc[(df['time'] > '16:15:00') & (df['time'] <= '16:45:00'), 'time'] = '16:30:00'
    df.loc[(df['time'] > '16:45:00') & (df['time'] <= '17:15:00'), 'time'] = '17:00:00'
    df.loc[(df['time'] > '17:15:00') & (df['time'] <= '17:45:00'), 'time'] = '17:30:00'
    df.loc[(df['time'] > '17:45:00'), 'time'] = '18:00:00'


    df = df.drop_duplicates(subset=['no', 'date', 'time'], keep='first', inplace=False)

    df.to_csv('2019_summer_clean.csv', index=False)
    print(df)
