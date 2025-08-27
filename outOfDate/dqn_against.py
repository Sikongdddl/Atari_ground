#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN (3-action head) self-play training on PettingZoo Atari Pong (pong_v3, AEC).
- 单模型、参数共享、左右对称：left agent 的观测做水平翻转
- 3 动作头：NOOP / UP / DOWN（训练更稳），推到环境前映射为真实动作 ID
- Double DQN + Reward Clipping ±1 + 稳健超参
- 自动发球状态机：仅在需要发球时强制 FIRE（不写回放），其余完全由模型决策
- 严格 AEC：done → step(None)，且终止步会写入回放
- TensorBoard 记录：loss/eps/buffer/q_mean/动作频率/回报/非零奖励比例
- 权重保存：latest / best（按 rolling_return_sum）/ final
"""

import os
import time
import random
from collections import deque, defaultdict, Counter
from typing import Dict, Deque, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.atari import pong_v3

# -----------------------------
# 超参
# -----------------------------
SEED = 0
STACK = 4
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_CAP = 200_000
TARGET_SYNC = 10_000
START_LEARN = 5_000
TOTAL_ENV_STEPS = 1_000_000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 1_000_000
LOG_EVERY = 1                 # 你要求更密日志
EVAL_EVERY = 50_000
RUN_NAME = "pz_pong_dqn3_doubledq"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 工具：帧预处理与堆叠
# -----------------------------
class FrameStacker:
    """为每个 agent 维护 4 帧堆叠（84x84 灰度），支持水平翻转。"""
    def __init__(self, stack: int = 4):
        self.stack = stack
        self.frames: Deque[np.ndarray] = deque(maxlen=stack)

    @staticmethod
    def _preprocess(frame: np.ndarray, flip: bool) -> np.ndarray:
        cut = frame[34:194, :, :]  # 裁顶部/底部信息条
        gray = cv2.cvtColor(cut, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        if flip:
            resized = cv2.flip(resized, 1)
        return (resized.astype(np.float32) / 255.0)[None, ...]  # [1,84,84]

    def prime_with(self, frame: np.ndarray, flip: bool):
        p = self._preprocess(frame, flip)
        self.frames.clear()
        for _ in range(self.stack):
            self.frames.append(p)

    def add(self, frame: np.ndarray, flip: bool):
        if len(self.frames) == 0:
            self.prime_with(frame, flip)
        else:
            self.frames.append(self._preprocess(frame, flip))

    def state(self) -> np.ndarray:
        while len(self.frames) < self.stack:
            self.frames.append(np.zeros((1, 84, 84), dtype=np.float32))
        return np.concatenate(list(self.frames), axis=0)  # [4,84,84]

    def reset(self):
        self.frames.clear()

# -----------------------------
# DQN（3动作头）
# -----------------------------
class DQN(nn.Module):
    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),  # 3: NOOP/UP/DOWN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)               # [B,64,7,7]
        x = x.view(x.size(0), -1)
        return self.fc(x)              # [B,3]

# -----------------------------
# 简单经验回放（uint8 存帧）
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, 4, 84, 84), dtype=np.uint8)
        self.act = np.zeros((capacity,), dtype=np.int64)      # 0/1/2（3 动作头）
        self.rew = np.zeros((capacity,), dtype=np.float32)    # 已剪裁 ±1
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool):
        self.obs[self.idx] = (np.clip(s, 0, 1) * 255).astype(np.uint8)
        self.next_obs[self.idx] = (np.clip(ns, 0, 1) * 255).astype(np.uint8)
        self.act[self.idx] = int(a)
        self.rew[self.idx] = float(r)
        self.done[self.idx] = float(d)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self.size >= batch_size, "replay size < batch_size"
        ids = np.random.randint(0, self.size, size=batch_size)

        s_np  = np.ascontiguousarray(np.asarray(self.obs[ids],      dtype=np.float32) / 255.0)
        ns_np = np.ascontiguousarray(np.asarray(self.next_obs[ids], dtype=np.float32) / 255.0)
        a_np  = np.asarray(self.act[ids],  dtype=np.int64)
        r_np  = np.asarray(self.rew[ids],  dtype=np.float32)
        d_np  = np.asarray(self.done[ids], dtype=np.float32)

        s  = torch.tensor(s_np,  dtype=torch.float32)
        ns = torch.tensor(ns_np, dtype=torch.float32)
        a  = torch.tensor(a_np,  dtype=torch.long)
        r  = torch.tensor(r_np,  dtype=torch.float32)
        d  = torch.tensor(d_np,  dtype=torch.float32)
        return s, a, r, ns, d

# -----------------------------
# 线性 ε 策略
# -----------------------------
def linear_eps(frame_idx: int) -> float:
    if frame_idx >= EPS_DECAY:
        return EPS_END
    t = frame_idx / EPS_DECAY
    return EPS_START + (EPS_END - EPS_START) * t

# -----------------------------
# 环境 & 动作映射（不依赖 meanings）
# -----------------------------
def make_env():
    env = pong_v3.env(render_mode=None)
    env.reset(seed=SEED)
    return env

def build_action_mapping(env) -> Tuple[int, int, int, int]:
    """
    返回： (NOOP_ID, UP_ID, DOWN_ID, FIRE_ID or -1)
    对 Pong 使用常见索引做 fallback：0:NOOP, 1:FIRE, 2:UP, 3:DOWN, 4:RIGHT, 5:LEFT
    同时做兜底：若动作数 < 6，尽可能推断 NOOP/UP/DOWN。
    """
    n = env.action_space(env.possible_agents[0]).n
    if n >= 6:
        NOOP_ID, FIRE_ID, UP_ID, DOWN_ID = 0, 1, 2, 3
        return NOOP_ID, UP_ID, DOWN_ID, FIRE_ID
    # 非常规构建兜底
    NOOP_ID = 0
    FIRE_ID = 1 if n >= 2 else -1
    if n >= 4:
        UP_ID, DOWN_ID = 2, 3
        return NOOP_ID, UP_ID, DOWN_ID, FIRE_ID
    # 最差兜底：UP/DOWN 不存在时临时指向 NOOP
    UP_ID, DOWN_ID = (2 if n > 2 else 0), (3 if n > 3 else 0)
    return NOOP_ID, UP_ID, DOWN_ID, FIRE_ID

# -----------------------------
# 训练主流程
# -----------------------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env = make_env()
    agents = list(env.possible_agents)
    # 动作映射（3→真实）
    NOOP_ID, UP_ID, DOWN_ID, FIRE_ID = build_action_mapping(env)
    # print(f"[Mapping] NOOP={NOOP_ID}, UP={UP_ID}, DOWN={DOWN_ID}, FIRE={FIRE_ID}, action_n={env.action_space(env.possible_agents[0]).n}")

    def map3_to_full(a3: int) -> int:
        return [NOOP_ID, UP_ID, DOWN_ID][int(a3)]

    # 网络（3 动作头）+ Double DQN 组件
    q = DQN(action_dim=3).to(DEVICE)
    q_tgt = DQN(action_dim=3).to(DEVICE)
    q_tgt.load_state_dict(q.state_dict())
    optim_ = torch.optim.RMSprop(q.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01, momentum=0.0, centered=False)

    # 回放、日志、保存
    rb = ReplayBuffer(REPLAY_CAP)
    writer = SummaryWriter(log_dir=os.path.join("runs", RUN_NAME))
    os.makedirs("checkpoints", exist_ok=True)

    # per-agent 帧堆叠与 transition 缓存
    stackers: Dict[str, FrameStacker] = {ag: FrameStacker(STACK) for ag in agents}
    last_state: Dict[str, np.ndarray] = {ag: None for ag in agents}
    last_action3: Dict[str, int] = {ag: None for ag in agents}

    # 统计
    ep_return = defaultdict(float)
    ep_len = defaultdict(int)
    episode_idx = 0  
    global_frame = 0
    best_eval = -1e9
    start_time = time.time()
    act_counter = Counter()
    # 非零奖励统计（滚动）
    trans_count = 0
    nonzero_reward_count = 0

    # 自动发球状态机
    serve_turn = False     # True->first_0 发，False->second_0 发
    need_serve = True      # 开局需要发球
    last_rewards_snapshot = {ag: 0.0 for ag in agents}

    env.reset()

    try:
        while global_frame < TOTAL_ENV_STEPS:
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                done = termination or truncation

                # —— 新一分检测（得分变化）——
                reward_changed = any(env.rewards[ag] != last_rewards_snapshot[ag] for ag in env.rewards)
                for ag in env.rewards:
                    last_rewards_snapshot[ag] = env.rewards[ag]
                if reward_changed:
                    serve_turn = not serve_turn
                    need_serve = True
                    for ag in stackers:
                        stackers[ag].reset()

                # ① 先推进帧栈，得到当前 state（即将作为上一拍 transition 的 next_state）
                flip = (agent == "first_0")
                stkr = stackers[agent]
                if len(stkr.frames) == 0:
                    stkr.prime_with(obs, flip)
                else:
                    stkr.add(obs, flip)
                cur_state = stkr.state()  # [4,84,84]

                # ② 若有上一次 (s,a3)：先补全上一拍 transition（注意这里使用 done 标记终止）
                if last_state[agent] is not None and last_action3[agent] is not None:
                    rew = float(np.sign(reward))  # reward clipping ±1
                    rb.add(last_state[agent], last_action3[agent], rew, cur_state, done)
                    trans_count += 1
                    nonzero_reward_count += int(abs(rew) > 0)
                    last_state[agent] = None
                    last_action3[agent] = None

                # ③ 如果 done：AEC 规范只能 step(None)，然后 reset 并继续
                if done:
                    env.step(None)
                    stkr.reset()
                    continue

                # 统计该 agent 回报（可选）
                ep_return[agent] += float(reward)
                ep_len[agent] += 1

                # ④ 自动发球（不写回放）
                if need_serve and FIRE_ID != -1:
                    serving_agent = "first_0" if serve_turn else "second_0"
                    if agent == serving_agent:
                        env.step(FIRE_ID)
                        need_serve = False
                    else:
                        env.step(NOOP_ID)
                    # 不缓存这一步（不进回放）
                    continue

                # ⑤ 选择动作（3 动作头）
                eps = linear_eps(global_frame)
                if (rb.size < START_LEARN) or (random.random() < eps):
                    a3 = random.randrange(3)  # 0/1/2
                else:
                    with torch.no_grad():
                        s_np = np.ascontiguousarray(np.asarray(cur_state, dtype=np.float32))
                        s = torch.tensor(s_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1,4,84,84]
                        qvals = q(s)          # [1,3]
                        a3 = int(qvals.argmax(dim=1).item())
                    act_counter[a3] += 1

                # 缓存当前 (s,a3)，等待下一次该 agent 轮到时补全
                last_state[agent] = cur_state
                last_action3[agent] = a3

                # ⑥ 映射到真实动作并推进环境
                env.step([NOOP_ID, UP_ID, DOWN_ID][a3])
                global_frame += 1

                # ⑦ 学习
                if (rb.size >= max(START_LEARN, BATCH_SIZE)) and (global_frame % 4 == 0):
                    bs, ba, br, bns, bd = rb.sample(BATCH_SIZE)
                    bs = bs.to(DEVICE); bns = bns.to(DEVICE)
                    ba = ba.to(DEVICE); br = br.to(DEVICE); bd = bd.to(DEVICE)

                    # Q(s,a)
                    q_sa = q(bs).gather(1, ba.view(-1, 1)).squeeze(1)
                    # Double DQN 目标
                    with torch.no_grad():
                        next_a = q(bns).argmax(dim=1, keepdim=True)       # 在线网选 a'
                        next_q = q_tgt(bns).gather(1, next_a).squeeze(1)  # 目标网估 Q(s',a')
                        target = br + (1.0 - bd) * GAMMA * next_q

                    loss = F.smooth_l1_loss(q_sa, target)
                    optim_.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    optim_.step()

                    # 日志
                    if global_frame % LOG_EVERY == 0:
                        writer.add_scalar("train/loss", loss.item(), global_frame)
                        writer.add_scalar("train/epsilon", eps, global_frame)
                        writer.add_scalar("train/buffer_size", rb.size, global_frame)
                        with torch.no_grad():
                            q_mean = q(bs).mean().item()
                        writer.add_scalar("train/q_mean", q_mean, global_frame)
                        # 动作频率
                        if act_counter:
                            total = sum(act_counter.values())
                            for k in [0, 1, 2]:
                                writer.add_scalar(f"policy/a{k}_freq", act_counter[k] / max(1, total), global_frame)
                            act_counter.clear()
                    # 非零奖励比例（基于写入回放的 transition）
                    if global_frame % 100*LOG_EVERY == 0 and trans_count > 0:
                        writer.add_scalar("train/nonzero_reward_ratio",
                                            nonzero_reward_count / trans_count, global_frame)
                        # 滚动窗口：清零重新统计
                        trans_count = 0
                        nonzero_reward_count = 0

                # 目标网同步
                if global_frame % TARGET_SYNC == 0:
                    q_tgt.load_state_dict(q.state_dict())

                # 简单 rolling sum 自评 + 保存
                if global_frame % EVAL_EVERY == 0:
                    total_ret = float(ep_return["first_0"] + ep_return["second_0"])
                    writer.add_scalar("eval/rolling_return_sum", total_ret, global_frame)
                    latest_path = os.path.join("checkpoints", f"{RUN_NAME}_latest.pth")
                    torch.save(q.state_dict(), latest_path)
                    if total_ret > best_eval:
                        best_eval = total_ret
                        best_path = os.path.join("checkpoints", f"{RUN_NAME}_best.pth")
                        torch.save(q.state_dict(), best_path)
                        print(f"[{global_frame}] New best rolling_return_sum={total_ret:.2f} -> {best_path}")

                if global_frame >= TOTAL_ENV_STEPS:
                    break

            # AEC 一轮结束；若无 agent 了，重置并清状态
            if not env.agents:
                # 1) 计算并写入 TensorBoard
                ep_first  = float(ep_return.get("first_0",  0.0))
                ep_second = float(ep_return.get("second_0", 0.0))
                ep_sum    = ep_first + ep_second
                ep_steps  = int(ep_len.get("first_0", 0) + ep_len.get("second_0", 0))

                writer.add_scalar("episode/return_first",  ep_first,  episode_idx)
                writer.add_scalar("episode/return_second", ep_second, episode_idx)
                writer.add_scalar("episode/return_sum",    ep_sum,    episode_idx)
                writer.add_scalar("episode/steps",         ep_steps,  episode_idx)

                # 2) 控制台打印
                print(f"[EP {episode_idx}] return_first={ep_first:.1f}  return_second={ep_second:.1f}  sum={ep_sum:.1f}  steps={ep_steps}")
                episode_idx += 1  # 自增 episode 计数

                env.reset()
                for ag in agents:
                    last_state[ag] = None
                    last_action3[ag] = None
                    stackers[ag].reset()
                ep_return.clear(); ep_len.clear()
                need_serve = True   # 新开一局需要发球

    finally:
        final_path = os.path.join("checkpoints", f"{RUN_NAME}_final.pth")
        torch.save(q.state_dict(), final_path)
        writer.close()
        env.close()
        print(f"Training done. Weights saved to: {final_path}")

if __name__ == "__main__":
    main()
