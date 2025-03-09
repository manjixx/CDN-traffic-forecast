import numpy as np
import tensorflow as tf
from keras import layers
from datetime import datetime, timedelta


# 环境类（添加时间约束和动态预测）
class VersionOptimizationEnv:
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


# Actor-Critic网络（TensorFlow实现）
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(action_dim, activation='tanh')  # 输出[-1,1]
        ])
        self.critic = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_ratio=0.2, gamma=0.99, lr=3e-4):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamma

    def act(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action_mean, _ = self.policy(state)
        return action_mean.numpy()[0]

    def update(self, states, actions, rewards, next_states):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)

        with tf.GradientTape() as tape:
            # 计算新旧策略概率
            new_means, values = self.policy(states)
            _, next_values = self.policy(next_states)

            # 计算优势
            advantages = rewards + self.gamma * next_values - values

            # PPO损失
            ratio = tf.exp(self._logprob(new_means, actions) - tf.stop_gradient(self._logprob(old_means, actions))
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Critic损失
            critic_loss = tf.reduce_mean(tf.square(rewards + self.gamma * next_values - values))

            total_loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(total_loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

    def _logprob(self, means, actions):
        # 计算对数概率（假设标准差为0.3）
        std = tf.ones_like(means) * 0.3
        dist = tfp.distributions.Normal(means, std)
        return dist.log_prob(actions)

# 创建推理专用类
class VersionOptimizer:
    def __init__(self, model_path):
        self.agent = PPOAgent(state_dim=5, action_dim=4)
        self.agent.load_model(model_path)
        self.env = None  # 需在实际使用时初始化

    def initialize_env(self, bandwidth_pred, versions, start_date):
        """初始化环境"""
        self.env = VersionOptimizationEnv(
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
