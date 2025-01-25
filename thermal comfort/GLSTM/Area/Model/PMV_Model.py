import numpy as np
from sklearn.metrics import *


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
            tcl_guess=0.5*(tcl_guess+tcl_cal)
        else:
            break

        if N > 150:
            break
    # print(N)
    # print(tcl_cal - tcl_guess)
    # print(tcl_cal)
    return tcl_cal, hc





