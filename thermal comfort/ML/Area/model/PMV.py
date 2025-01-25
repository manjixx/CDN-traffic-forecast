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
    df = pd.read_csv('../dataset/synthetic.csv').dropna(axis=0, how='any', inplace=False)
    # 季节
    m = 1.1
    clo = 0.6
    va = 0.2
    data = np.array(df[['ta', 'hr']].drop_duplicates(subset=['ta', 'hr'], keep='first', inplace=False))

    # 初始化存储列表
    pmv_pred = []
    result = []

    for i in range(0, len(data)):
        res = []
        ta = data[i][0]
        rh = data[i][1]
        # vel =
            # data[i][2]
        res.append(ta)
        res.append(rh)

        pmv_result = pmv_model(M=m * 58.15, clo=clo, tr=ta, ta=ta, vel=va, rh=rh)
        pmv_pred.append(round(pmv_result, 2))

        if pmv_result > 0.5:
            res.append(2)
        elif pmv_result < -0.5:
            res.append(0)
        else:
            res.append(1)
        result.append(res)
    print(result)
    pd.DataFrame(columns=['ta', 'hr', 'tsv'], data=result).to_csv(f'../PLOT/dataset/pmv.csv', index=False)
