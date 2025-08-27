import os
import gym
from gym import spaces
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
# 定义经验回放存储结构
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """深度Q网络结构"""
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 输入通道=4帧堆叠
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PongWrapper(gym.Wrapper):
    """Atari Pong环境预处理"""
    def __init__(self, env, stack_frames=4):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        
        # Pong专用动作映射（实际有效动作只有3个）
        self.valid_actions = [0, 2, 3]  # [无操作, 上, 下]
        
        self.action_space = spaces.Discrete(len(self.valid_actions))  # 动作空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(stack_frames, 84, 84), dtype=np.float32
        )  # 状态空间

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(0)  # 初始空操作
        processed = self._preprocess(obs)
        for _ in range(self.stack_frames):
            self.frames.append(processed)
        return self._stack_frames()
    
    def step(self, action):
        # 将DQN选择的动作映射到实际有效动作
        real_action = self.valid_actions[action]
        
        total_reward = 0
        for _ in range(4):  # Frame skipping: 每4帧执行一次动作
            obs, reward, done, info = self.env.step(real_action)
            total_reward += reward
            processed = self._preprocess(obs)
            self.frames.append(processed)
            if done:
                break
        return self._stack_frames(), total_reward, done, info
    
    def _preprocess(self, frame):
        # 裁剪+灰度化+下采样
        frame = frame[34:194, :, :]  # 裁剪计分板区域
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame / 255.0  # 归一化
    
    def _stack_frames(self):
        return torch.FloatTensor(np.stack(self.frames))

    def close(self):
        return self.env.close()
# 训练参数配置
config = {
    "env_name": "PongNoFrameskip-v4",
    "batch_size": 64,
    "gamma": 0.99,
    "lr": 1e-4,
    "memory_size": 100000,
    "update_target": 5,  # 更新目标网络的间隔
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 1000,
    "max_episodes": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def train():
    # 初始化环境
    base_env = gym.make(config["env_name"])
    env = PongWrapper(base_env)
    
    # 初始化网络
    policy_net = DQN(len(env.valid_actions)).to(config["device"])
    target_net = DQN(len(env.valid_actions)).to(config["device"])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    memory = ReplayMemory(config["memory_size"])
    
    epsilon = config["epsilon_start"]
    episode_rewards = []
    
    for episode in range(config["max_episodes"]):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # epsilon-贪婪策略选择动作
            if random.random() < epsilon:
                action = random.randint(0, len(env.valid_actions)-1)
            else:
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0).to(config["device"])
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 存储经验
            memory.push(state, action, next_state, reward, done)
            
            state = next_state
            
            # 训练步骤
            if len(memory) >= config["batch_size"]:
                transitions = memory.sample(config["batch_size"])
                batch = Transition(*zip(*transitions))
                
                # 转换数据为张量
                state_batch = torch.stack(batch.state).to(config["device"])
                action_batch = torch.LongTensor(batch.action).view(-1, 1).to(config["device"])
                reward_batch = torch.FloatTensor(batch.reward).to(config["device"])
                next_state_batch = torch.stack(batch.next_state).to(config["device"])
                done_batch = torch.FloatTensor(batch.done).to(config["device"])
                
                # 计算当前Q值
                current_q = policy_net(state_batch).gather(1, action_batch)
                
                # 计算目标Q值
                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1)[0]
                    target_q = reward_batch + (1 - done_batch) * config["gamma"] * next_q
                
                # 计算损失
                loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
                
                # 优化步骤
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(),max_norm=1.0)
                optimizer.step()
                
        # 更新目标网络
        if episode % config["update_target"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 衰减epsilon
        epsilon = max(config["epsilon_end"], 
                     config["epsilon_start"] - episode / config["epsilon_decay"])
        
        # 记录训练数据
        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode: {episode}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    # 保存模型和关闭TensorBoard
    torch.save(policy_net.state_dict(), "pong_dqn.pth")

    print("训练完成！模型已保存为 pong_dqn.pth")

# if __name__ == "__main__":
#     # train()
#     base_env = gym.make(config["env_name"])
#     env = PongWrapper(base_env)
#     env = gym.wrappers.Monitor(env,'./video',force=True)
#     state = env.reset()
#     # 初始化网络
#     policy_net = DQN(len(env.valid_actions))
#     policy_net.load_state_dict(torch.load("./pong_dqn.pth"))

#     for _ in range(2100):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         action_probs = policy_net(state_tensor)
#         action = torch.argmax(action_probs).item()

#         state,reward,done ,_ = env.step(action)

#         if done:
#             state = env.reset()
#     env.close()

import csv
import time

finetune_config = {
    # —— 训练控制 —— 
    "env_name": "PongNoFrameskip-v4",
    "pretrained_path": "pong_dqn.pth",       # 载入已有权重
    "save_dir": "./finetune_ckpts",          # 权重与CSV保存目录
    "epochs": 1,                             # 微调多少个epoch
    "steps_per_epoch": 20000,                # 每个epoch训练多少步（environment steps）
    "eval_episodes": 5,                      # 每个epoch后评测多少局

    # —— 超参（与原config保持风格一致）——
    "batch_size": 64,
    "gamma": 0.99,
    "lr": 5e-5,                              # 微调可用更小LR
    "memory_size": 100000,
    "target_update_interval": 1000,          # 按步数更新target（微调期间更稳）
    "clip_grad_norm": 1.0,

    # —— ε-greedy（微调期独立调度）——
    "epsilon_start": 0.99,                   
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 50000,            # 按步数线性衰减

    # —— 复现实验 —— 
    "seed": 123,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def make_env_oldgym(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)        # 旧API里是 env.seed
        except Exception:
            pass
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return PongWrapper(env)

@torch.no_grad()
def evaluate(env, policy_net, n_episodes=5, device="cpu"):
    """老Gym API评测：返回 avg_return 与 avg_length。"""
    policy_net.eval()
    returns = []
    lengths = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            state_tensor = state.unsqueeze(0).to(device)  # (1,4,84,84)
            q = policy_net(state_tensor)
            action = q.argmax(dim=1).item()
            next_state, reward, done, _ = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            state = next_state
        returns.append(ep_ret)
        lengths.append(ep_len)
    policy_net.train()
    return float(np.mean(returns)), float(np.mean(lengths))

def linear_epsilon_by_step(step, start, end, decay_steps):
    if step >= decay_steps:
        return end
    # 线性从 start → end
    return start + (end - start) * (step / max(1, decay_steps))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def finetune(cfg=finetune_config):
    ensure_dir(cfg["save_dir"])

    # —— 环境 —— 
    env = make_env_oldgym(cfg["env_name"], seed=cfg["seed"])

    # —— 网络（加载已有权重）——
    action_dim = len(env.valid_actions)
    policy_net = DQN(action_dim).to(cfg["device"])
    target_net = DQN(action_dim).to(cfg["device"])

    if not os.path.isfile(cfg["pretrained_path"]):
        raise FileNotFoundError(f"预训练权重不存在: {cfg['pretrained_path']}")
    policy_net.load_state_dict(torch.load(cfg["pretrained_path"], map_location=cfg["device"]))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    memory = ReplayMemory(cfg["memory_size"])

    # —— 记录CSV —— 
    csv_path = os.path.join(cfg["save_dir"], "finetune_metrics.csv")
    csv_header = [
        "epoch", "train_steps_cum", "epsilon",
        "last_100_avg_reward", "eval_avg_return", "eval_avg_len",
        "loss_avg_epoch", "time_sec_epoch"
    ]
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    # —— 训练状态 —— 
    global_step = 0
    episode_rewards_window = deque(maxlen=100)  # 近100局平均
    loss_accum = 0.0
    loss_count = 0
    last_target_sync = 0

    # —— 开始训练 —— 
    for epoch in range(1, cfg["epochs"] + 1):
        start_time = time.time()
        steps_this_epoch = 0
        # 每个epoch从一个新episode开始
        state = env.reset()
        done = False
        ep_ret = 0.0

        while steps_this_epoch < cfg["steps_per_epoch"]:
            # ε-贪婪（按step衰减）
            epsilon = linear_epsilon_by_step(
                global_step,
                cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay_steps"]
            )
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q = policy_net(state.unsqueeze(0).to(cfg["device"]))
                    action = int(q.argmax(dim=1).item())

            next_state, reward, done, _ = env.step(action)

            # 存入回放（保持类型与原训练一致：state/next_state都是Tensor）
            memory.push(state, action, next_state, float(reward), bool(done))

            state = next_state
            ep_ret += float(reward)
            steps_this_epoch += 1
            global_step += 1

            # 一局结束则重置
            if done:
                episode_rewards_window.append(ep_ret)
                state = env.reset()
                done = False
                ep_ret = 0.0

            # 采样并更新
            if len(memory) >= cfg["batch_size"]:
                transitions = memory.sample(cfg["batch_size"])
                batch = Transition(*zip(*transitions))

                state_batch = torch.stack(batch.state).to(cfg["device"])
                action_batch = torch.LongTensor(batch.action).view(-1, 1).to(cfg["device"])
                reward_batch = torch.FloatTensor(batch.reward).to(cfg["device"])
                next_state_batch = torch.stack(batch.next_state).to(cfg["device"])
                done_batch = torch.FloatTensor(batch.done).to(cfg["device"])

                # Q(s,a)
                current_q = policy_net(state_batch).gather(1, action_batch).squeeze(1)

                # y = r + (1-d) * gamma * max_a' Q_target(s', a')
                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1)[0]
                    target_q = reward_batch + (1.0 - done_batch) * cfg["gamma"] * next_q

                loss = nn.SmoothL1Loss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                if cfg["clip_grad_norm"] is not None:
                    nn.utils.clip_grad_norm_(policy_net.parameters(), cfg["clip_grad_norm"])
                optimizer.step()

                loss_accum += float(loss.item())
                loss_count += 1

            # 固定步数同步 target（微调更稳定）
            if global_step - last_target_sync >= cfg["target_update_interval"]:
                target_net.load_state_dict(policy_net.state_dict())
                last_target_sync = global_step

        # —— 一个epoch结束：评测 & 记录 —— 
        avg_reward_100 = float(np.mean(episode_rewards_window)) if len(episode_rewards_window) > 0 else 0.0
        eval_avg_ret, eval_avg_len = evaluate(
            make_env_oldgym(cfg["env_name"], seed=cfg["seed"] + epoch),
            policy_net, n_episodes=cfg["eval_episodes"], device=cfg["device"]
        )
        loss_avg = (loss_accum / max(1, loss_count))
        time_spent = time.time() - start_time

        # 打印
        print(f"[Epoch {epoch}/{cfg['epochs']}] "
              f"steps+={steps_this_epoch} (total {global_step}) | "
              f"eps={epsilon:.3f} | "
              f"loss_avg={loss_avg:.5f} | "
              f"eval_ret={eval_avg_ret:.2f} len={eval_avg_len:.1f} | "
              f"last100_avg_reward={avg_reward_100:.2f} | "
              f"{time_spent:.1f}s")

        # CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, global_step, round(epsilon, 6),
                round(avg_reward_100, 4), round(eval_avg_ret, 4), round(eval_avg_len, 2),
                round(loss_avg, 6), round(time_spent, 2)
            ])

        # 保存权重（按epoch）
        ckpt_path = os.path.join(cfg["save_dir"], f"pong_dqn_ft_epoch{epoch}.pth")
        torch.save(policy_net.state_dict(), ckpt_path)

        # reset epoch内统计
        loss_accum = 0.0
        loss_count = 0

    print(f"微调完成。权重与统计已保存到：{cfg['save_dir']}")
    return policy_net

if __name__ == "__main__":
    # 运行微调
    finetune()