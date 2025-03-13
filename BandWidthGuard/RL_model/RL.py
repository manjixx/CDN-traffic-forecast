import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, optimizers, losses
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from .configs import RLConfig

# Actor 网络
class Actor(keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = keras.Sequential([
            layers.Dense(64, activation="tanh", input_shape=state_dim),  # 输入层到隐藏层
            layers.Dense(action_dim),   # 隐藏层到输出层
            layers.Softmax(axis=-1)     # 输出层应用Softmax
        ])
    def call(self, state):
        # 前向传播，返回动作概率
        return self.actor(state)

# Critic 网络（）
class Critic(keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 网络：状态价值评估
        self.critic = keras.Sequential([
            layers.Dense(64, input_shape=state_dim, activation='tanh'), # 输入层到隐藏层
            layers.Dense(1) # 输出状态价值（线性层）
        ])

    def call(self, state):
        # 前向传播，返回动作分布概率
        return self.critic(state)

class PPO:
    def __int__(self,state_dim, action_dim, config):
        self.config = config
        self.actor = Actor(state_dim, action_dim)   # 初始化Actor网络
        self.critic = Critic(state_dim) # 初始化 Critic 网络
        self.actor_optimizer = optimizers.Adam(self.actor.parameters(), lr = config.lr)  # Actor 优化器
        self.critic_optimizer = optimizers.Adam(self.critic.parameters(), lr = config.lr)  # Critic 优化器
        # self.mse_loss = losses.mean_squared_error()

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)     # 转换为状态张量
        action_probs = self.actor(state)    # 获取动作概率
        dist = tfp.distributions.Categorical(probs = action_probs)  # 创建类别分布
        action = dist.sample()  # 采样动作
        action_log_prob = dist.log_prob(action) # 获取动作的对数概率
        return action.item, action_log_prob.item    # 返回动作和对数概率

    def compute_age(self, values, rewards, dones, next_value):
        advantages = []
        age = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value   # 最后一步的下一个值
            else:
                next_val = values[t + 1]   # 其他步骤的下一个值

            delta = rewards[t] + self.config.gamma * (next_val * (1 - dones[t])) - values[t] # TD 误差计算
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * age # 计算 GAE
            advantages.insert(0, gae)   # 插入到优势列表开头

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        return advantages

    def update(self, states, actions, old_log_probs, rewards, dones, next_state):
        # 转换为 TensorFlow 张量
        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)   # 转换状态为张量
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)             # 转换动作为张量
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)   # 转换旧的对数概率为张量
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # 计算 GAE
        with tf.stop_gradient:
            values = tf.squeeze(self.critic(states))    # 获取当前状态的价值
            next_value = tf.squeeze(self.critic(next_state))    # 获取下一个状态的价值

        values_np = values.numpy()  # 为了兼容原有计算方式
        advantages = self.compute_gae(values_np.tolist(), rewards, dones, next_value.numpy())   # 计算优势
        returns = advantages + values   # 计算回报，即优势+状态价值，因为advantages是基于TD误差的是相对值，所以需要加上状态价值得到绝对值，来稳定训练
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)    # 标准化优势


        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, old_log_probs, advantages, returns)
        ).shuffle(len(states)).batch(self.config.batch_size)

        # PPO 多轮更新
        for _ in range(self.config.ppo_epochs):
            for batch in dataset:
                # 获取小批量状态、获取小批量动作、获取小批量旧的对数概率、获取小批量优势、获取小批量回报
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

                with tf.GradientTape(persistent=True) as tape:
                    # 计算新的动作概率
                    action_probs = self.actor(batch_states)
                    dist = tfp.distributions.Categorical(probs=action_probs)
                    new_log_probs = dist.log_prob(batch_actions)

                    # 计算比率和裁剪后的目标
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.config.clip_epsilon,
                                             1 + self.config.clip_epsilon) * batch_advantages
                    # 计算熵
                    entropy = tf.reduce_mean(dist.entropy())

                    # 计算Actor损失（加入熵正则化项）
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.config.entropy_coef * entropy

                    # 计算 Critic 损失
                    critic_values = tf.squeeze(self.critic(batch_states))
                    critic_loss = self.mse_loss(critic_values, batch_returns)

                # 更新 Actor
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

                # 更新 Critic
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                del tape  # 清除持久梯度带

    def train_ppo(sel):
        config = RLConfig()  # 初始化配置
        # env = gym.make("CartPole-v1")  # 创建环境
        state_dim = env.observation_space.shape[0]  # 获取状态维度
        action_dim = env.action_space.n  # 获取动作维度

        agent = PPO(state_dim, action_dim, config)  # 初始化PPO智能体

        rewards = []  # 用于存储每个回合的奖励 (保持相同)

        for episode in range(config.max_episodes):
            state = env.reset()[0]  # 重置环境
            states, actions, rewards_buffer, log_probs, dones = [], [], [], [], []  # 初始化存储变量 (保持相同)
            total_reward = 0  # 初始化总奖励

            for step in range(config.max_steps):
                action, log_prob = agent.get_action(state)  # 获取动作和对数概率
                next_state, reward, done, _, _ = env.step(
                    action.numpy()[0] if tf.is_tensor(action) else action)  # 执行动作

                states.append(state)  # 存储状态
                actions.append(action.numpy() if tf.is_tensor(action) else action)  # 存储动作
                rewards_buffer.append(reward)  # 存储奖励
                log_probs.append(log_prob.numpy() if tf.is_tensor(log_prob) else log_prob)  # 存储对数概率
                dones.append(done)  # 存储是否结束

                total_reward += reward  # 累加奖励
                state = next_state  # 更新状态

                if done:
                    break


            agent.update(
                states,
                actions,
                log_probs,
                rewards,
                dones,
                next_state  # 最后一个next_state已经是numpy数组
            )  # 更新智能体 (内部实现改为TF)

            rewards.append(total_reward)  # 记录每个回合的奖励
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")  # 打印当前回合的奖励

            if total_reward >= 500:
                print("Solved!")  # 终止条件
                break

        env.close()  # 关闭环境

        # 绘制reward曲线
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()


# 环境类（添加时间约束和动态预测）
class VersionEnv:
    def __init__(self, initial_bandwidth_pred, existing_versions, new_versions, start_date):
        self.current_bandwidth = initial_bandwidth_pred  # 初始预测
        self.existing_versions = existing_versions
        self.new_versions = new_versions
        self.time_interval = 5  # 分钟
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # 预计算有效时间点（排除周末）
        self.valid_timesteps = self._precompute_valid_timesteps(len(initial_bandwidth_pred))

    def _precompute_valid_timesteps(self, total_steps):
        """生成有效时间点索引（排除周末）"""
        valid = []
        for t in range(total_steps):
            current_time = self.start_date + timedelta(minutes=t * 5)
            if current_time.weekday() < 5:  # 0-4表示周一到周五
                valid.append(t)
        return np.array(valid)

    def reset(self):
        # 初始化新版本参数（记录原始发布时间）
        for v in self.new_versions:
            v['original_time'] = v['release_time']
            v['release_time'] = v['original_time']  # 重置为原始时间
            v['release_ratio'] = 0.0
        return self._get_state()

    def _get_state(self):
        # 状态包含当前带宽特征和版本信息
        stats = [
            np.mean(self.current_bandwidth),
            np.percentile(self.current_bandwidth, 95),
            sum(v['release_ratio'] for v in self.new_versions),
            len(self.new_versions),
            (self.new_versions[0]['release_time'] - self.new_versions[0]['original_time']) / 300  # 标准化时间偏移
        ]
        return np.array(stats, dtype=np.float32)

    def _is_valid_time(self, timestep):
        """检查是否为有效发布时间"""
        return timestep in self.valid_timesteps

    def _find_next_valid_time(self, timestep):
        """找到下一个有效时间点"""
        idx = np.searchsorted(self.valid_timesteps, timestep)
        if idx < len(self.valid_timesteps):
            return self.valid_timesteps[idx]
        return timestep  # 如果超出范围保持原值

    def step(self, action):
        # 解析动作 [时间偏移1, 比例1, 时间偏移2, 比例2,...]
        for i in range(len(self.new_versions)):
            # 处理时间偏移（只能向后调整）
            time_offset = int(np.clip(action[2 * i], 0, 1) * 288)  # 最大调整24小时（288个5分钟）
            original_t = self.new_versions[i]['original_time']
            new_t = original_t + time_offset

            # 有效性检查
            if not self._is_valid_time(new_t):
                new_t = self._find_next_valid_time(new_t)

            self.new_versions[i]['release_time'] = new_t

            # 处理比例（sigmoid）
            ratio = 1 / (1 + np.exp(-action[2 * i + 1]))
            self.new_versions[i]['release_ratio'] = ratio

        # 归一化比例
        total = sum(v['release_ratio'] for v in self.new_versions)
        if total > 1e-3:
            for v in self.new_versions:
                v['release_ratio'] /= total

        # 调用预测模型获取新带宽（这里需要接入实际预测接口）
        updated_bandwidth = self._call_prediction_model()
        self.current_bandwidth = updated_bandwidth

        # 计算奖励
        sorted_bw = np.sort(updated_bandwidth)
        cost = sorted_bw[int(0.95 * len(sorted_bw))]
        reward = -cost

        return self._get_state(), reward, True, {}

    def _call_prediction_model(self):
        """实际需要接入预测服务，这里用随机生成示例"""
        return self.current_bandwidth * np.random.uniform(0.9, 1.1, len(self.current_bandwidth))

# 创建推理专用类
class VersionOptimizer:
    def __init__(self, model_path):
        self.agent = PPO(state_dim=5, action_dim=4)
        self.agent.load_model(model_path)
        self.env = None  # 需在实际使用时初始化

    def initialize_env(self, bandwidth_pred, versions, start_date):
        """初始化环境"""
        self.env = VersionEnv(
            initial_bandwidth_pred=bandwidth_pred,
            existing_versions=[],
            new_versions=versions,
            start_date=start_date
        )

    def get_optimized_plan(self, current_state=None):
        """获取优化策略"""
        if not current_state:
            state = self.env.reset()
        else:
            state = current_state

        action = self.agent.act(state)
        optimized_versions = []

        # 解析动作
        for i in range(len(self.env.new_versions)):
            time_offset = int(np.clip(action[2 * i], 0, 1) * 288)
            original_t = self.env.new_versions[i]['original_time']
            new_t = original_t + time_offset
            ratio = 1 / (1 + np.exp(-action[2 * i + 1]))

            # 记录优化结果
            optimized_versions.append({
                'version_name': self.env.new_versions[i]['name'],
                'release_time': self._convert_timestep(new_t),
                'release_ratio': float(ratio)
            })

        return {
            'bandwidth_95th': -self.env.step(action)[1],  # 获取奖励值转换回带宽
            'versions': optimized_versions
        }

    def _convert_timestep(self, timestep):
        """将时间步转换为可读格式"""
        total_mins = timestep * 5
        days = total_mins // 1440
        hours = (total_mins % 1440) // 60
        mins = total_mins % 60
        return f"{days}天{hours:02d}:{mins:02d}"
# 训练示例
if __name__ == "__main__":
    # 初始化环境参数
    start_date = "2024-03-01"
    initial_bandwidth = np.random.lognormal(5, 0.2, 8640)  # 示例数据
    existing_versions = []
    new_versions = [
        {'package_size': 500, 'in_network_users': 1e4, 'release_time': 0},
        {'package_size': 300, 'in_network_users': 8e3, 'release_time': 1440}  # 初始时间（分钟）
    ]

    env = VersionOptimizationEnv(initial_bandwidth, existing_versions, new_versions, start_date)
    agent = PPOAgent(state_dim=5, action_dim=4)

    # 训练参数
    episodes = 1000
    batch_size = 32

    # 训练循环
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0

        # 执行单步（版本发布是单次决策）
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # 存储经验
        agent.update(
            states=tf.expand_dims(state, 0),
            actions=tf.expand_dims(action, 0),
            rewards=tf.expand_dims(reward, 0),
            next_states=tf.expand_dims(next_state, 0)
        )

        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {reward:.2f}, 95th Bandwidth: {-reward:.2f} Mbps")

            agent.save_model(f"models/checkpoint_ep{ep}")

            # 最终保存
        agent.save_model("models/final_model")

    # 输出最终策略
    final_action = agent.act(env.reset())
    print("\nOptimized Release Strategy:")
    for i, v in enumerate(new_versions):
        print(f"Version {i + 1}:")
        print(
            f"  Original Time: {v['original_time'] // 1440}d {(v['original_time'] % 1440) // 60:02d}:{(v['original_time'] % 60):02d}")
        print(
            f"  Optimized Time: {v['release_time'] // 1440}d {(v['release_time'] % 1440) // 60:02d}:{(v['release_time'] % 60):02d}")
        print(f"  Release Ratio: {v['release_ratio']:.2%}")

    # 使用示例
    optimizer = VersionOptimizer("models/final_model")

    # 准备输入数据
    bandwidth_pred = [...]  # 从预测模型获取的带宽数据
    new_versions = [
        {'name': 'v3.2.1', 'package_size': 500,
         'in_network_users': 1e4, 'release_time': 0},
        {'name': 'v3.2.2', 'package_size': 300,
         'in_network_users': 8e3, 'release_time': 1440}
    ]

    # 初始化环境
    optimizer.initialize_env(
        bandwidth_pred=bandwidth_pred,
        versions=new_versions,
        start_date="2024-03-01"
    )

    # 获取优化方案
    plan = optimizer.get_optimized_plan()
    print("优化后的带宽峰值(95%):", plan['bandwidth_95th'], "Mbps")
    for v in plan['versions']:
        print(f"版本{v['version_name']}:")
        print(f"  发布时间: {v['release_time']}")
        print(f"  发布比例: {v['release_ratio']:.2%}")

# https://www.cnblogs.com/GreenOrange/articles/18582769
