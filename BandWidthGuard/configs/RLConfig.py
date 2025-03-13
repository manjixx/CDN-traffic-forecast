class RLConfig:

    clip_epsilon = 0.2  # PPO裁剪系数，用于限制策略更新的幅度
    ppo_epochs = 10  # 每次更新的PPO轮数
    batch_size = 64  # 小批量训练的大小
    gae_lambda = 0.95  # GAE参数，用于计算优势函数

    # 基础参数
    hidden_dim = 64  # 神经网络隐藏层维度
    lr = 0.0003  # 学习率
    gamma = 0.99  # 折扣因子
    max_episodes = 5000  # 最大训练回合数
    max_steps = 500  # 每个回合最大步数
    seed = 42  # 随机种子
    entropy_coef = 0.01  # 熵正则化系数，用于鼓励探索
