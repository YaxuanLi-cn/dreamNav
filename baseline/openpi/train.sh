WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ours \
  --exp-name=pi05_ours \
  --data.data-dir /root/dreamNav/pairUAV/train \
  --data.tour-dir /root/dreamNav/pairUAV/tours \
  --overwrite
  