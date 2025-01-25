import os
import moxing as mox
if not os.path.exists('./network_traffic_predict'):
    mox.file.copy('obs://modelarts-labs-bj4-v2/course/modelarts/zjc_team/time_series_forecast/network_traffic_forecast/network_traffic_forecast.zip', './network_traffic_forecast.zip')
    os.system('unzip network_traffic_forecast.zip')

if not os.path.exists('./network_traffic_forecast'):
    raise Exception('错误！数据不存在！')
