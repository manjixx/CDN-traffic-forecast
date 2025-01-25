"""预测使用特征，对应训练与预测输入数据列名，list格式"""
load_feature_name=['p']                        #负荷预测所使用的特征
load_future_feature_name=None                    #负荷预测未来预报特征，如未来时刻天气预报等，如果无该特征，设置为None
PV_feature_name=['p', 'illumination', 'temperature', 'humidity'] #光伏预测使用的特征
PV_future_feature_name=['temperature']                   #光伏预测未来预报特征，如未来时刻天气预报等，如果无该特征，设置为None


"""训练程序输入：训练数据集路径"""
cold_load_train_data_path = './Data/Train/cold_load_train.xls'          #冷
ele_load_train_data_path = './Data/Train/ele_load_train.xls'            #电
heat_load_train_data_path = './Data/Train/heat_load_train.xls'          #热
PV_train_data_path = './Data/Train/PV_train.xls'                        #光伏

"""训练程序输出/预测程序输入：模型保存路径"""
#冷负荷实时4h预测与24h预测模型
cold_load_4h_model_path = './Model/cold_load_4h.pt'
cold_load_24h_model_path = './Model/cold_load_24h.pt'
#热负荷实时4h预测与24h预测模型
heat_load_4h_model_path = './Model/heat_load_4h.pt'
heat_load_24h_model_path = './Model/heat_load_24h.pt'
#电负荷实时4h预测与24h预测模型
ele_load_4h_model_path = './Model/ele_load_4h.pt'
ele_load_24h_model_path = './Model/ele_load_24h.pt'
#光伏实时4h预测与24h预测模型
PV_4h_model_path = './Model/PV_4h.pt'
PV_24h_model_path = './Model/PV_24h.pt'

"""预测程序输入：预测输入数据路径"""
#冷负荷实时4h预测与24h预测输入模型
cold_load_4h_pre_data_path = './Data/Predict/cold_load_4h_pre.xls'
cold_load_24h_pre_data_path = './Data/Predict/cold_load_24h_pre.xls'
#热负荷
heat_load_4h_pre_data_path = './Data/Predict/heat_load_4h_pre.xls'
heat_load_24h_pre_data_path = './Data/Predict/heat_load_24h_pre.xls'
#电负荷
ele_load_4h_pre_data_path = './Data/Predict/ele_load_4h_pre.xls'
ele_load_24h_pre_data_path = './Data/Predict/ele_load_24h_pre.xls'
#光伏
PV_4h_pre_data_path = './Data/Predict/PV_4h_pre.xls'
PV_24h_pre_data_path = './Data/Predict/PV_24h_pre.xls'

"""预测程序输出：预测结果保存路径"""
#冷负荷实时4h预测与24h预测结果
cold_load_4h_result_data_path = './Data/Result/cold_load_4h_pre_result.xls'
cold_load_24h_result_data_path = './Data/Result/cold_load_24h_pre_result.xls'
#热负荷
heat_load_4h_result_data_path = './Data/Result/heat_load_4h_pre_result.xls'
heat_load_24h_result_data_path = './Data/Result/heat_load_24h_pre_result.xls'
#电负荷
ele_load_4h_result_data_path = './Data/Result/ele_load_4h_pre_result.xls'
ele_load_24h_result_data_path = './Data/Result/ele_load_24h_pre_result.xls'
#光伏
PV_4h_result_data_path = './Data/Result/PV_4h_pre_result.xls'
PV_24h_result_data_path = './Data/Result/PV_24h_pre_result.xls'


