import os, time, random, math
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo.atari import pong_v3
import supersuit as ss

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ======================
# 1) 环境 & 动作映射
# ======================
def make_env():
    env = pong_v3.parallel_env(obs_type='grayscale_image')
    # 典型像素预处理
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, np.float32)        
    env = ss.normalize_obs_v0(env, 0, 255)     
    return env

class ActionMapper:
    """
    把 PZ 的 legal_actions 映射到 {NOOP, UP, DOWN} 三动作集合，
    通过查询而非硬编码，避免 Atari 版本差异。
    """
    def __init__(self, env, agent_id):
        env.reset()
        legal = env.legal_actions[agent_id] if hasattr(env, "legal_actions") else env.action_spaces[agent_id].n
        # fallback：部分版本没有 legal_actions 属性，这里直接根据 actions.n 来构造映射
        if isinstance(legal, list):
            acts = legal
        else:
            acts = list(range(legal))
        # 尝试自动找 NOOP/UP/DOWN：最朴素做法——保留 0/2/3（Atari Pong 常见）
        # 如果你已有准确映射，替换这里
        cand = { "NOOP": 0, "UP": 2, "DOWN": 3 }
        for k,v in tuple(cand.items()):
            if v not in acts:
                # 回退：任选三个
                cand = { "NOOP": acts[0], "UP": acts[1], "DOWN": acts[2] }
                break
        self.map = cand
        self.inv = [cand["NOOP"], cand["UP"], cand["DOWN"]]
    @property
    def n(self): return 3
    def to_env(self, a_small:int) -> int:
        return self.inv[a_small]

# ======================
# 2) DQN 网络 & Agent
# ======================
class DuelingDQN(nn.Module):
    def __init__(self, in_ch=4, n_actions=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        # 84x84 -> 7x7 after conv strides
        self.fc = nn.Linear(64*7*7, 512)
        self.val = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)
    def forward(self, x):  # x: (B,4,84,84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        v = self.val(x)
        a = self.adv(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

Transition = namedtuple("Transition", "s a r s2 d")

class ReplayBuffer:
    def __init__(self, capacity=500_000):
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)

    def add(self, s, a, r, s2, d):
        self.buf.append(Transition(
            torch.as_tensor(s,  dtype=torch.uint8).contiguous(),
            int(a),
            float(r),
            torch.as_tensor(s2, dtype=torch.uint8).contiguous(),
            float(d),
        ))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        # 直接 torch.stack，不走 numpy
        s  = torch.stack([b.s  for b in batch], dim=0).to(torch.float32)
        a  = torch.tensor([b.a  for b in batch], dtype=torch.long)
        r  = torch.tensor([b.r  for b in batch], dtype=torch.float32)
        s2 = torch.stack([b.s2 for b in batch], dim=0).to(torch.float32)
        d  = torch.tensor([b.d  for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

class DQNAgent:
    def __init__(self, n_actions, lr=1e-4, gamma=0.99, device="cuda"):
        self.q = DuelingDQN(4, n_actions).to(device)
        self.t = DuelingDQN(4, n_actions).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
    @torch.no_grad()
    def act(self, obs_chw, eps:float, eval_mode=False):
        if (not eval_mode) and random.random() < eps:
            return random.randrange(self.q.adv.out_features)

        x = torch.as_tensor(obs_chw, dtype=torch.float32, device=self.device)  # (1,4,84,84)
        if x.ndim == 3: 
            x = x.unsqueeze(0)
        q = self.q(x)
        return int(q.argmax(dim=1).item())
    def update(self, batch, target_update_tau=None, huber=True, clip=10.0):
        s, a, r, s2, d = [x.to(self.device) if torch.is_tensor(x) else x for x in batch]
        with torch.no_grad():
            # Double DQN
            a2 = self.q(s2).argmax(dim=1, keepdim=True)              # (B,1)
            q_t = self.t(s2).gather(1, a2).squeeze(1)                # (B,)
            y = r + (1.0 - d) * self.gamma * q_t
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y) if huber else F.mse_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), clip)
        self.opt.step()
        if target_update_tau is None:
            return float(loss.item())
        # 软更新
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.t.parameters()):
                tp.data.mul_(1 - target_update_tau).add_(target_update_tau * p.data)
        return float(loss.item())
    def hard_update(self): self.t.load_state_dict(self.q.state_dict())
    def save(self, path):
        torch.save(self.q.state_dict(), path)
    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.q.load_state_dict(sd); self.hard_update()

# ======================
# 3) 对手（脚本/冻结/池）
# ======================
class OpponentBase:
    def reset(self): pass
    def act(self, obs): raise NotImplementedError

class RandomOpponent(OpponentBase):
    def __init__(self, n_actions=3): self.n = n_actions
    def act(self, obs): return random.randrange(self.n)

class ScriptedOpponent(OpponentBase):
    """极简启发式：估个球的垂直趋势 -> 上/下/不动（像素下仅示意，稳定性尚可）"""
    def __init__(self):
        self.prev_y = None
    def reset(self): self.prev_y = None
    def act(self, obs):
        # obs: (4,84,84) -> 取最后一帧估个 y 质心
        last = obs[-1]
        col = last.sum(axis=1)  # over x
        y = int(col.argmax())
        if self.prev_y is None:
            self.prev_y = y; return 0  # NOOP
        dy = y - self.prev_y; self.prev_y = y
        if dy > 1: return 2   # DOWN
        if dy < -1: return 1  # UP
        return 0

class FrozenOpponent(OpponentBase):
    def __init__(self, weight_path, device="cuda"):
        self.agent = DQNAgent(3, device=device); self.agent.load(weight_path)
    @torch.no_grad()
    def act(self, obs): return self.agent.act(obs, eps=0.0, eval_mode=True)

class SnapshotPool(OpponentBase):
    def __init__(self, paths, device="cuda", recent_bias=0.7):
        self.paths = list(paths)
        self.device = device
        self.recent_bias = recent_bias
        self._cache = {}
    def _load(self, p):
        if p not in self._cache:
            a = DQNAgent(3, device=self.device); a.load(p); a.q.eval()
            self._cache[p] = a
        return self._cache[p]
    @torch.no_grad()
    def act(self, obs):
        if not self.paths:
            return random.randrange(3)
        # 近期偏置：几何分布抽样
        idx = int(len(self.paths) - 1 - np.random.geometric(1 - self.recent_bias))
        idx = max(0, min(idx, len(self.paths)-1))
        a = self._load(self.paths[idx])
        return a.act(obs, eps=0.0, eval_mode=True)
    def add(self, p): self.paths.append(p)

def serve_once(env, my_id, opp_id):
    try:
        env.step({my_id: 1, opp_id: 0})  # 1=FIRE, 0=NOOP
    except Exception:
        pass  # 某些版本如果已在进行中会抛错，忽略即可

# ======================
# 4) 训练（只训练一侧）
# ======================
def train_one_side(side="left",
                   total_steps=500000,
                   warmup=10_000,
                   eval_every=1000,
                   save_every=100_000,
                   batch_size=128,
                   lr=1e-4,
                   gamma=0.99,
                   eps_start=1.0,
                   eps_end=0.05,
                   eps_decay_steps=500000,
                   target_tau=0.005,
                   device="cuda",
                   opponent_kind="scripted",
                   frozen_path=None,
                   snapshot_pool_paths=None,
                   save_dir="checkpoints"):
    """
    side: 'left' -> 训练 first_0；'right' -> 训练 second_0
    opponent_kind: 'scripted' | 'random' | 'frozen' | 'pool'
    """
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(
        "runs",
        f"pz_pong_{side}_{opponent_kind}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TB] Logging to: {log_dir}")

    env = make_env()
    agents = env.possible_agents  # ["first_0", "second_0"]
    my_id   = "first_0"  if side == "left"  else "second_0"
    opp_id  = "second_0" if side == "left"  else "first_0"

    env.reset()
    amap_my  = ActionMapper(env, my_id)
    amap_opp = ActionMapper(env, opp_id)

    # 对手
    if opponent_kind == "scripted": opponent = ScriptedOpponent()
    elif opponent_kind == "random": opponent = RandomOpponent(amap_opp.n)
    elif opponent_kind == "frozen":
        assert frozen_path is not None
        opponent = FrozenOpponent(frozen_path, device)
    elif opponent_kind == "pool":
        opponent = SnapshotPool(snapshot_pool_paths or [], device=device)
    else:
        raise ValueError("unknown opponent")

    agent = DQNAgent(amap_my.n, lr=lr, gamma=gamma, device=device)
    rb = ReplayBuffer(500_000)

    def eps_by_step(t):
        ratio = min(1.0, t / float(eps_decay_steps))
        return eps_start + (eps_end - eps_start) * ratio

    obs, infos = env.reset()
    serve_once(env, my_id, opp_id)
    # --- episode stats for my side ---
    ep_ret = 0.0
    ep_len = 0
    episode_idx = 0

    for k in range(total_steps):
        if k % 100 == 0:
            print("at step", k)
        if my_id not in obs:  # 某些版本 reset 后要再次 reset
            obs, infos = env.reset()

        # 取我方与对手观测（CHW）
        ob_my  = np.transpose(obs[my_id], (2,0,1))
        ob_opp = np.transpose(obs[opp_id], (2,0,1))

        eps = eps_by_step(k) if k >= warmup else 1.0
        a_my_small  = agent.act(ob_my, eps, eval_mode=False)
        a_opp_small = opponent.act(ob_opp)

        a_env = {
            my_id:  amap_my.to_env(a_my_small),
            opp_id: amap_opp.to_env(a_opp_small),
        }
                
        next_obs, rews, terminations, truncations,infos = env.step(a_env)
        dones = {a: terminations.get(a, False) or truncations.get(a, False) for a in rews.keys()}
        # 只存我方 transition
        ob2_my = np.transpose(next_obs[my_id], (2,0,1))
        r_my   = float(rews[my_id])
        d_my   = bool(dones[my_id])
        rb.add(ob_my, a_my_small, r_my, ob2_my, float(d_my))
        # --- episode stats (my side) ---
        ep_ret += r_my
        ep_len += 1
        if d_my:
            writer.add_scalar("episode/return", ep_ret, episode_idx)
            writer.add_scalar("episode/length", ep_len, episode_idx)
            episode_idx += 1
            ep_ret = 0.0
            ep_len = 0

        obs = next_obs
        if any(dones.values()):
            opponent.reset()
            obs, infos = env.reset()
            serve_once(env, my_id, opp_id)

        # 学习
        if len(rb) >= max(warmup, batch_size):
            batch = rb.sample(batch_size)
            loss = agent.update(batch, target_update_tau=target_tau)
            # --- TB: training scalars ---
            writer.add_scalar("train/loss", loss, k)
            writer.add_scalar("train/epsilon", eps, k)
            writer.add_scalar("train/replay_size", len(rb), k)
            # 学习率（可能是多组 param_group，这里记录第一个）
            writer.add_scalar("train/lr", agent.opt.param_groups[0]["lr"], k)
        
        # 日志/评估/保存
        if (k+1) % eval_every == 0:
            win = evaluate_simple(env, agent, opponent, my_id, opp_id, amap_my, amap_opp, device)
            print(f"[{side}] step={k+1} eval_winrate={win:.3f} eps={eps:.3f}")
            writer.add_scalar("eval/winrate", win, k+1)      
        if (k+1) % save_every == 0:
            path = os.path.join(save_dir, f"{side}_step{k+1}.pth")
            agent.save(path)
            print(f"saved: {path}")
            if isinstance(opponent, SnapshotPool):
                opponent.add(path)  # 把我方快照加入池，供另一条训练线使用

    # 最终保存
    agent.save(os.path.join(save_dir, f"{side}_final.pth"))
    writer.flush()
    writer.close()

# ======================
# 5) 简单评估（单边固定对手）
# ======================
@torch.no_grad()
def evaluate_simple(env, my_agent, opponent, my_id, opp_id, amap_my, amap_opp, device):
    wins = 0
    for _ in range(episodes):
        print("eval episode", _)
        obs, infos = env.reset()
        opponent.reset()

        done = {a: False for a in env.agents}
        score = {my_id:0.0, opp_id:0.0}

        while not any(done.values()):
            ob_my  = np.transpose(obs[my_id], (2,0,1))
            ob_opp = np.transpose(obs[opp_id], (2,0,1))
            a_my   = my_agent.act(ob_my, eps=0.0, eval_mode=True)
            a_op   = opponent.act(ob_opp)
            obs, rews, terminations, truncations, infos = env.step({
                my_id: amap_my.to_env(a_my),
                opp_id: amap_opp.to_env(a_op)
            })
            done = {a: terminations.get(a, False) or truncations.get(a, False) for a in rews.keys()}
            score[my_id] += float(rews[my_id])
            score[opp_id] += float(rews[opp_id])
        wins += 1 if score[my_id] > score[opp_id] else 0
    return wins / float(episodes)

if __name__ == "__main__":
    # 示例：先训左侧（对手脚本），得到 left_final.pth；再把它当 frozen，对右侧开训
    #train_one_side(side="left",  opponent_kind="scripted", total_steps=500_000, device="cuda")
    # 然后：train_one_side(side="right", opponent_kind="frozen", frozen_path="checkpoints/left_final.pth",
    #                     total_steps=500_000, device="cuda")
    train_one_side(side="right",opponent_kind="scripted", total_steps=500_000, device="cuda")