"""
包括数据处理等辅助函数
"""
import numpy as np
import pandas as pd


def data_processing(data, window_size, predict_size,step=1,feature_feats=None):
    """
    :param data: 要处理的原始data，DataFrame
    :param window_size: 滑动窗口长度，即lstm输入序列的长度
    :param predict_size: 预测序列的长度
    :param step: 滑动窗口每次平移的间隔
    :param feature_feats: 未来特征的名称list格式，如果没有，默认为None
    :return: 一个dict，包含处理后的历史特征输入、输出、未来特征输入
    """
    # 标准化
    data = (data - data.mean()) / data.std()
    data_in = []
    data_out = []
    data_future_feats = []

    for i in range(0, data.shape[0] - window_size - predict_size, step):
        data_in.append(np.array(data[i:i + window_size]))
        data_out.append(np.array(data['p'][i + window_size:i + window_size + predict_size]))
        if feature_feats:
            data_future_feats.append(np.array(data[feature_feats][i + window_size:i + window_size + predict_size]))
    data_in = np.array(data_in)
    data_out = np.array(data_out)
    if feature_feats:
        data_future_feats = np.array(data_future_feats)
    else:
        data_future_feats = data_out
    data_process = {'data_in': data_in, 'data_out': data_out, 'data_future_feats': data_future_feats}
    return data_process

