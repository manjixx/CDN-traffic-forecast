import os
import sys
import json
from typing import Any, Dict, Tuple
import keras_tuner as kt
import tensorflow as tf
from keras import callbacks
from keras.optimizers import Optimizer

# 获取当前文件的绝对路径
current_path = os.path.abspath(__file__)
# 获取项目根目录
root_path = os.path.dirname(os.path.dirname(current_path))
# 将根目录添加到Python路径中
sys.path.append(root_path)
from configs.hyper_params import SearchSpace, ModelConfig
from configs.data_config import DataConfig


class HyperparameterTuner:
    def __init__(self, search_space: SearchSpace, model):
        self.search_space = search_space
        self.base_model = model  # 存储基础模型架构
        self.tuner: kt.Tuner = None
        self.best_hps: kt.HyperParameters = None
        self.best_model: tf.keras.Model = None

    def _build_hypermodel(self, hp: kt.HyperParameters) -> tf.keras.Model:

        """从 SearchSpace 采样生成 ModelConfig"""
        # Transformer 参数采样
        d_model = hp.Choice('d_model', self.search_space.d_model_range)
        nhead = hp.Choice('nhead', self.search_space.n_head_option)
        num_transformer_layers = hp.Choice('trans_layers', self.search_space.num_transformer_layers_range)

        # LSTM 参数采样
        num_lstm_layers = hp.Int('num_lstm_layers', *self.search_space.lstm_layers_range)
        hidden_dims = [
            hp.Int(f'hidden_{i}_dims',
                   min_value=self.search_space.hidden_dims_range[0],
                   max_value=self.search_space.hidden_dims_range[1],
                   step=self.search_space.hidden_dims_range[2])
            for i in range(num_lstm_layers)
        ]

        # 公共参数采样
        runtime_params = {
            'dropout_rate': hp.Float('dropout', *self.search_space.dropout_range, step=0.1),
            'attention_units': hp.Int('attn_units', *self.search_space.attention_units_range),
            'learning_rate': hp.Float('lr', *self.search_space.learning_rate_range, sampling='log'),
            # 'batch_size': hp.Choice('batch_size', self.search_space.batch_size),
            'optimizer': hp.Choice('optimizer', self.search_space.optimizer_options)
        }

        # 构建完整配置
        config = ModelConfig(
            d_model=d_model,
            n_head=nhead,
            num_transformer_layers=num_transformer_layers,
            num_lstm_layers=num_lstm_layers,
            hidden_dims=tuple(hidden_dims),
            **runtime_params
        )

        # 实例化并编译模型
        model = self.base_model(config)
        model.compile(loss='mse', metrics=['mae'])
        return model

    def _sample_hyperparameters(self, hp: kt.HyperParameters) -> Dict[str, Any]:
        """从搜索空间采样参数"""
        params = {}

        # 遍历搜索空间定义
        for param, config in self.search_space.param_configs.items():
            param_type = config["type"]

            try:
                if param_type == "int":
                    params[param] = hp.Int(
                        param,
                        min_value=config["min"],
                        max_value=config["max"],
                        step=config.get("step", 1)
                    )
                elif param_type == "float":
                    params[param] = hp.Float(
                        param,
                        min_value=config["min"],
                        max_value=config["max"],
                        step=config.get("step", None),
                        sampling=config.get("sampling", "linear")
                    )
                elif param_type == "category":
                    params[param] = hp.Choice(
                        param,
                        values=config["values"]
                    )
                elif param_type == "bool":
                    params[param] = hp.Boolean(
                        param,
                        default=config.get("default", False)
                    )
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            except KeyError as e:
                raise ValueError(f"Missing required config for {param}: {e}")

        # 处理依赖参数
        if "lstm_layers" in params:
            # 从 HIDDEN_DIM 中选择对应的隐藏层维度
            hidden_dims_options = self.search_space.HIDDEN_DIM
            lstm_layers = params["lstm_layers"]

            # 确保选择的层数不超过 HIDDEN_DIM 的选项数量
            if lstm_layers <= len(hidden_dims_options):
                params["hidden_dims"] = hidden_dims_options[lstm_layers - 1]  # 选择对应的隐藏层维度
            else:
                raise ValueError(f"lstm_layers must be less than or equal to {len(hidden_dims_options)}")

        return params

    def _build_optimizer(self, hp: kt.HyperParameters) -> Optimizer:
        """构建优化器"""
        opt_name = hp.Choice(
            "OPTIMIZER",
            self.search_space.param_configs["OPTIMIZER"]["values"]
        )
        lr = hp.Float(
            "learning_rate",
            min_value=self.search_space.param_configs["learning_rate"]["min"],
            max_value=self.search_space.param_configs["learning_rate"]["max"],
            sampling="log"
        )

        optimizers = {
            "adam": tf.keras.optimizers.Adam,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "sgd": tf.keras.optimizers.SGD
        }

        if opt_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # 处理优化器特有参数
        kwargs = {}
        if opt_name == "sgd":
            kwargs["momentum"] = hp.Float(
                "sgd_momentum",
                min_value=0.0,
                max_value=0.9,
                step=0.1
            )

        return optimizers[opt_name](learning_rate=lr, **kwargs)

    def tune(
            self,
            train_data: tf.data.Dataset,
            val_data: tf.data.Dataset,
            algorithm: str = "bayesian",
            max_trials: int = 50,
            executions_per_trial: int = 2,
            objective: str = "val_loss",
            project_name: str = "hptune",
            **kwargs
    ) -> kt.HyperParameters:
        """
        执行超参数搜索

        :param train_data: 训练数据集 (features, labels)
        :param val_data: 验证数据集 (features, labels)
        :param algorithm: 调优算法 ['bayesian', 'random', 'hyperband']
        :param max_trials: 最大试验次数
        :param executions_per_trial: 每个超参数组合的验证次数
        :param objective: 优化目标
        :param project_name: 项目名称
        """
        # 初始化调优器
        tuner_map = {
            "bayesian": kt.BayesianOptimization,
            "random": kt.RandomSearch,
            "hyperband": kt.Hyperband
        }

        self.tuner = tuner_map[algorithm.lower()](
            hypermodel=self._build_hypermodel,
            objective=kt.Objective(objective, direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=self.search_space.output_dir,
            project_name=project_name,
            overwrite=True
        )

        # 添加数据格式验证
        self._validate_data_shape(train_data)
        self._validate_data_shape(val_data)

        # 配置回调函数
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=objective,
                patience=self.search_space.early_stopping_patience,
                restore_best_weights=True
            ),

            callbacks.CSVLogger(f"{project_name}_log.csv"),
            callbacks.TensorBoard(log_dir=f"{self.search_space.log_dir}/{project_name}")
        ]

        # 执行搜索
        self.tuner.search(
            x=train_data,  # 直接传入生成器/Dataset
            validation_data=val_data,
            callbacks=callbacks_list,
            **kwargs
        )

        # 获取最优参数
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"🏆 Optimal hyperparameters: {json.dumps(self.best_hps.values, indent=2)}")
        return self.best_hps

    def train_best_model(
            self,
            train_data: Tuple[tf.data.Dataset, tf.data.Dataset],
            val_data: Tuple[tf.data.Dataset, tf.data.Dataset],
            epochs: int = 200
    ) -> tf.keras.callbacks.History:
        """使用最优参数训练最终模型"""
        if not self.best_hps:
            raise ValueError("请先执行tune方法进行参数搜索")

        # 构建最终模型
        self.best_model = self.tuner.hypermodel.build(self.best_hps)

        # 配置生产环境回调
        checkpoint = callbacks.ModelCheckpoint(
            f"{self.search_space.model_dir}/best_model.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False
        )

        # 执行训练
        history = self.best_model.fit(
            x=train_data[0],
            y=train_data[1],
            validation_data=val_data,
            epochs=epochs,
            callbacks=[
                checkpoint,
                callbacks.CSVLogger(f"{self.search_space.log_dir}/final_training.csv"),
                callbacks.TensorBoard(f"{self.search_space.log_dir}/final")
            ],
            verbose=1
        )
        return history

    def save_assets(self, version: str = "v1") -> None:
        """保存全套生产资产"""
        if not self.best_model:
            raise ValueError("未找到训练完成的模型")

        # 创建保存目录
        save_dir = f"{self.search_space.model_dir}/{version}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型
        self.best_model.save(f"{save_dir}/model.h5")

        # 保存参数配置
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump({
                "hyperparameters": self.best_hps.values,
                "fixed_parameters": self.search_space.fixed_params
            }, f, indent=2)

        print(f"✅ 生产资产保存到：{save_dir}")

    def _validate_data_shape(self, data):
        """验证数据格式"""
        sample_data = next(iter(data))
        if not isinstance(sample_data, (tuple, list)) or len(sample_data) != 2:
            raise ValueError("数据生成器必须返回(features, labels)元组")
