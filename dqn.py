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

if __name__ == "__main__":
    # train()
    base_env = gym.make(config["env_name"])
    env = PongWrapper(base_env)
    env = gym.wrappers.Monitor(env,'./video',force=True)
    state = env.reset()
    # 初始化网络
    policy_net = DQN(len(env.valid_actions))
    policy_net.load_state_dict(torch.load("./pong_dqn.pth"))

    for _ in range(2100):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = torch.argmax(action_probs).item()

        state,reward,done ,_ = env.step(action)

        if done:
            state = env.reset()
    env.close()