import torch
import cv2
import numpy as np
from collections import deque
from pettingzoo.atari import pong_v3
import torch.nn as nn

# ----------------- 模型 -----------------
class DQN(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------- 预处理 -----------------
class FrameProcessor:
    def __init__(self, stack_frames=4):
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)

    @staticmethod
    def _preprocess(frame, flip=False):
        # 裁掉顶部/底部信息条 -> 灰度 -> 84x84 -> 可选水平翻转 -> [0,1]
        frame = frame[34:194, :, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        if flip:
            frame = cv2.flip(frame, 1)
        return frame.astype(np.float32) / 255.0

    def add_frame(self, frame, flip=False):
        processed = self._preprocess(frame, flip=flip)
        self.frames.append(processed)

    def prime_with(self, frame, flip=False):
        processed = self._preprocess(frame, flip=flip)
        self.frames.clear()
        for _ in range(self.stack_frames):
            self.frames.append(processed)

    def get_state(self):
        while len(self.frames) < self.stack_frames:
            self.frames.append(np.zeros((84, 84), dtype=np.float32))
        # [4,84,84] -> torch[1,4,84,84]
        state_np = np.stack(self.frames)
        state_t = torch.tensor(
            np.ascontiguousarray(state_np, dtype=np.float32),
            dtype=torch.float32
        ).unsqueeze(0)
        return state_t

    def reset(self):
        self.frames.clear()

# ----------------- Letterbox 保持 4:3 -----------------
def letterbox(img_rgb, target_w, target_h):
    h, w = img_rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# ----------------- 主流程 -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先建环境，取真实动作数
    env = pong_v3.env(render_mode="rgb_array")
    env.reset()
    action_dim = env.action_space(env.possible_agents[0]).n  # e.g., 6

    # 构建与权重一致的模型（action_dim 必须匹配）
    model = DQN(action_dim=action_dim).to(device)
    state_dict = torch.load(
        "/home/ubuntu/jrCode/Atari_ground/checkpoints/pz_pong_selfplay_dqn_best.pth",
        map_location=device
    )
    model.load_state_dict(state_dict)  # 尺寸将匹配
    model.eval()
    print("Model loaded successfully with action_dim =", action_dim)

    processors = {ag: FrameProcessor(4) for ag in env.possible_agents}
    last_rewards = {agent: 0 for agent in env.possible_agents}

    # 录屏（保持4:3比例）
    save_video = True
    video_filename = "pong_gameplay_fast.avi"
    output_frame_rate = 60
    frame_skip = 1
    width, height = 480, 360

    video_writer = None
    written_frames = 0
    raw_frame_counter = 0

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(video_filename, fourcc, output_frame_rate, (width, height))
    with torch.no_grad():
        try:
            while True:
                for agent in env.agent_iter():
                    obs, reward, termination, truncation, info = env.last()
                    current_agent = env.agent_selection
                    done = termination or truncation

                    # 录屏：只在一个固定 agent 的回合抓帧，防止计数翻倍
                    if save_video and current_agent == "second_0":
                        frame = env.render()
                        if frame is not None:
                            if raw_frame_counter % frame_skip == 0:
                                # RGB -> BGR for OpenCV, then letterbox to 480x360
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                frame_bgr = letterbox(frame_bgr, width, height)
                                video_writer.write(frame_bgr)
                                written_frames += 1
                            raw_frame_counter += 1

                    # 必须在 done 时 step(None)（AEC 规范）
                    if done:
                        env.step(None)
                        processors[current_agent].reset()
                        # 刷新得分记录
                        for ag in env.rewards:
                            last_rewards[ag] = env.rewards[ag]
                        continue

                    # 处理输入（左侧翻转）
                    flip_input = (current_agent == "first_0")
                    processor = processors[current_agent]
                    if len(processor.frames) == 0:
                        processor.prime_with(obs, flip=flip_input)
                    else:
                        processor.add_frame(obs, flip=flip_input)
                    state = processor.get_state().to(device)  # [1,4,84,84]

                    # 模型决策：直接输出环境动作（动作维度=env 的真实动作数）
                    q_values = model(state)
                    action = int(q_values.argmax(dim=1).item())
                    env.step(action)

                if not env.agents:
                    break
        finally:
            if video_writer is not None:
                video_writer.release()
                video_duration = written_frames / float(output_frame_rate) if output_frame_rate > 0 else 0.0
                print(f"Video frames written: {written_frames}")
                print(f"Video duration: ~{video_duration:.1f} seconds")
                print(f"Video saved to {video_filename}")

    env.close()
    print("Game finished")

if __name__ == "__main__":
    main()
