# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：PMV.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/1 16:46 
"""
from PMV_Model import *
import numpy as np
import pandas as pd
import random

if __name__ == '__main__':
    # df = pd.read_csv('../SDataSet/Synthetic/pmv.csv').dropna(axis=0, how='any', inplace=False)
    df = pd.read_csv('./pmv.csv').dropna(axis=0, how='any', inplace=False)
    # 季节
    season = ['summer', 'winter']
    m = 1.1

    for s in season:
        print(s)
        if s == 'summer':
            data = df.loc[df['season'] == 0]
            env = data[['ta', 'hr']]
            clo = 0.5
        else:
            data = df.loc[df['season'] == 1]
            env = data[['ta', 'hr']]
            clo = 1.1
        data = np.array(data[['ta', 'hr']])

        # 初始化存储列表
        pmv_pred = []
        result = []

        for i in range(0, len(data)):
            res = []
            ta = data[i][0]
            rh = data[i][1]
            # vel =
            vel = 0.06
                # data[i][2]
            res.append(ta)
            res.append(rh)

            pmv_result = pmv_model(M=m * 58.15, clo=clo, tr=ta, ta=ta, vel=vel, rh=rh)
            pmv_pred.append(round(pmv_result, 2))

            if pmv_result > 0.5:
                res.append(2)
            elif pmv_result < -0.5:
                res.append(0)
            else:
                res.append(1)
            result.append(res)
        pd.DataFrame(columns=['ta', 'hr', 'tsv'], data=result).to_csv(f'../PLOT/dataset/{s}p.csv', index=False)
