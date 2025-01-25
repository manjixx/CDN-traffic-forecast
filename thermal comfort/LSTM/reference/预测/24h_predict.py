from config import *
from model import predict

if __name__ == '__main__':
    #电负荷24h预测
    predict(predict_data_path = cold_load_24h_pre_data_path,
            result_path = cold_load_24h_result_data_path,
            model_path = cold_load_24h_model_path,
            feature_name = load_feature_name,
            predict_length = 24,
            hidden_size = 128)

    #热负荷24h预测
    predict(predict_data_path = heat_load_24h_pre_data_path,
            result_path = heat_load_24h_result_data_path,
            model_path = heat_load_24h_model_path,
            feature_name = load_feature_name,
            predict_length = 24,
            hidden_size = 128)

    #冷负荷24h预测
    predict(predict_data_path = ele_load_24h_pre_data_path,
            result_path = ele_load_24h_result_data_path,
            model_path = ele_load_24h_model_path,
            feature_name = load_feature_name,
            predict_length = 24,
            hidden_size = 128)

    #光伏24h预测
    predict(predict_data_path = PV_24h_pre_data_path,
            result_path = PV_24h_result_data_path,
            model_path = PV_24h_model_path,
            feature_name = PV_feature_name,
            predict_length = 24,
            hidden_size = 128)