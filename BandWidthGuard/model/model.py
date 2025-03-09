import os
import sys
import tensorflow as tf
import keras_tuner as kt
from keras import layers, callbacks
# 选择一种导入方式（推荐使用 TensorFlow 的 Keras）
from keras_hub.layers import TransformerEncoder
from typing import List, Optional

# 路径设置（保持不变）
current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)
from configs.hyper_params import ModelConfig


class PositionalEncoding(layers.Layer):
    """可选项：为Transformer添加位置编码"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model

        # 初始化位置编码矩阵
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) *
                          (-tf.math.log(10000.0) / self.d_model))

        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(self.max_len, self.d_model),
            initializer='zeros',
            trainable=True  # 是否允许训练
        )

        self.pe[:, 0::2].assign(tf.sin(position * div_term))
        self.pe[:, 1::2].assign(tf.cos(position * div_term))
        self.pe = self.pe[tf.newaxis, :, :]  # (1, max_len, d_model)

    def call(self, x):
        # x的形状应为 (batch, time_steps, num_versions, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


class VersionTransformer(tf.keras.Model):
    def __init__(self, d_model, nhead, num_layers, version_feature_dim, use_pos_enc=False):
        super().__init__()
        self.d_model = d_model
        if d_model % 2 != 0:
            raise ValueError("d_model 必须是偶数")

        self.use_pos_enc = use_pos_enc
        self.version_feature_dim = version_feature_dim

        # 特征嵌入
        self.embed = layers.Dense(
            d_model,
            input_shape=(None, None, version_feature_dim)  # (N, Fv)
        )

        # 位置编码（可选）
        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        self.encoders = [
            TransformerEncoder(
                d_model,
                nhead,
                activation='relu',
                dropout=0.1,
                # batch_first=True
            ) for _ in range(num_layers)
        ]

        # CLS标记
        self.cls_token = tf.Variable(
            tf.random.normal((1, 1, self.d_model)),
            trainable=True
        )

    def call(self, inputs):
        # inputs: [B, num_versions, 4]
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        num_versions = tf.shape(inputs)[2]

        # 保持静态形状可追踪
        static_batch = inputs.shape[0] or batch_size
        static_time = inputs.shape[1] or time_steps

        # 展平时间维度
        x_flat = tf.reshape(
            inputs,
            [static_batch * static_time, num_versions, self.version_feature_dim]
        )

        x_embed = self.embed(x_flat)  # [B, N, d_model]

        if self.use_pos_enc:
            x_embed = self.pos_encoder(x_embed)

        # 拼接CLS标记
        # cls_tokens = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        new_batch_size = static_batch * static_time
        cls_tokens = tf.tile(self.cls_token.shape, [new_batch_size,1,1])
        x_embed = tf.concat([cls_tokens, x_embed], axis=1)

        # 通过编码器
        for encoder in self.encoders:
            x_embed = encoder(x_embed)
        output = tf.reshape(x_embed[:, 0, :], [batch_size, time_steps, self.d_model])

        return output  # [B, d_model]


class CombinedModel(tf.keras.Model):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.optimizer = config.optimizer

        # Transformer模块
        self.transformer = VersionTransformer(
            d_model=config.d_model,
            nhead=config.n_head,
            num_layers=config.num_transformer_layers,
            use_pos_enc=config.use_pos_enc,
            version_feature_dim=config.version_feature_dim
        )

        # LSTM配置
        self.num_lstm_layers = config.num_lstm_layers
        self.hidden_dims: List[int] = config.hidden_dims
        self.dropout_rate: float = config.dropout_rate
        self.use_attention = config.use_attention

        # 构建LSTM层
        self.lstm_layers = []
        self.dropout_layers = []

        for i in range(self.num_lstm_layers):  # Loop through number of layers
            # Use the corresponding hidden dimension for the current layer
            dim = self.hidden_dims[i] if i < len(self.hidden_dims) else self.hidden_dims[-1]

            # Return sequences for all layers except the last one unless use_attention is True
            return_sequences = (i != self.num_lstm_layers - 1) or self.config.use_attention

            self.lstm_layers.append(
                layers.LSTM(
                    dim,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                )
            )

            # Add Dropout layer after each LSTM layer except the last one
            if i < self.num_lstm_layers - 1:  # No dropout after the last layer
                self.dropout_layers.append(layers.Dropout(self.dropout_rate))

        # 注意力机制
        if self.use_attention:
            self.attention_dense = layers.Dense(config.attention_units, activation='tanh')

        # 输出层
        self.output_layer = layers.Dense(config.OUTPUT_SIZE)

    def call(self, inputs, training=None):
        fixed_seq, variable_seq = inputs[0], inputs[1]

        trans_out = self.transformer(variable_seq)
        combined = tf.concat([
            trans_out,
            fixed_seq
        ], axis=2)

        x = combined
        # 通过LSTM层
        for i, lstm in enumerate(self.lstm_layers):
            x = lstm(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x, training=training)

        # 注意力机制
        if self.use_attention:
            # 生成注意力分数 [B, T, 1]
            scores = self.attention_dense(x)
            scores = tf.squeeze(scores, axis=-1)  # [B, T]
            attention_weights = tf.nn.softmax(scores, axis=1)  # [B, T]

            # 计算上下文向量
            context = tf.reduce_sum(
                x * tf.expand_dims(attention_weights, -1),
                axis=1
            )  # [B, features]
        else:
            context = x  # 直接使用最后输出

        return self.output_layer(context)

