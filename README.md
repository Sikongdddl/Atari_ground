# Atari ground

## Atari rom set up:
pip install atari-py
python -m atari_py.import_roms <path to folder>

## shadow leaderborad usage
python shadow_leaderboard.py \
  --env PongNoFrameskip-v4 \
  --bank-size 100 \
  --csv ./shadow_leaderboard.csv \
  --model baseline=./pong_dqn.pth


python shadow_leaderboard.py \
  --env PongNoFrameskip-v4 \
  --csv ./shadow_leaderboard.csv \
  --model baseline=./pong_dqn.pth \
  --model ft_ep1=./finetune_ckpts/pong_dqn_ft_epoch1.pth \
  --model ft_ep2=./finetune_ckpts/pong_dqn_ft_epoch2.pth

## requirements:
opencv-python==4.10.0.82
python=3.9
torch==2.5.1 cuda11.8 cudnn9.1.0
gym=0.15.7
ale_py=0.11.2