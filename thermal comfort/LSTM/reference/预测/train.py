from config import *
from model import train

if __name__ == '__main__':
    #冷负荷实时4h预测模型训练
    train(cold_load_train_data_path, cold_load_4h_model_path, load_feature_name,
          sequence_length=24, predict_length=4, max_epoch=50, hidden_size=64)
    #冷负荷实时24h预测模型训练
    train(cold_load_train_data_path, cold_load_24h_model_path, load_feature_name,
          sequence_length=24*3, predict_length=24, max_epoch=100, hidden_size=128)
    #热负荷实时4h预测模型训练
    train(heat_load_train_data_path, heat_load_4h_model_path, load_feature_name,
          sequence_length=24, predict_length=4, max_epoch=50, hidden_size=64)
    #热负荷实时24h预测模型训练
    train(heat_load_train_data_path, heat_load_24h_model_path, load_feature_name,
          sequence_length=24*3, predict_length=24, max_epoch=100, hidden_size=128)
    #电负荷实时4h预测模型训练
    train(ele_load_train_data_path, ele_load_4h_model_path, load_feature_name,
          sequence_length=24, predict_length=4, max_epoch=50, hidden_size=64)
    #电负荷实时24h预测模型训练
    train(ele_load_train_data_path, ele_load_24h_model_path, load_feature_name,
          sequence_length=24*3, predict_length=24, max_epoch=100, hidden_size=128)
    #光伏负荷实时4h预测模型训练
    train(PV_train_data_path, PV_4h_model_path, PV_feature_name,
          sequence_length=24, predict_length=4, max_epoch=50, hidden_size=64)
    #光伏负荷实时24h预测模型训练
    train(PV_train_data_path, PV_24h_model_path, PV_feature_name,
          sequence_length=24*3, predict_length=24, max_epoch=100, hidden_size=128)