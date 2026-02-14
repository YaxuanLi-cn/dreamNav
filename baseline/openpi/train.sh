WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ours \
  --exp-name=pi05_ours \
  --data.data-dir /root/dreamNav/pairUAV/try_train \
  --data.tour-dir /root/dreamNav/pairUAV/tours \
  --overwrite
# Training auto-stops after 1 epoch (try_train ≈ 364 steps, full train ≈ 255k steps).
# Switch --data.data-dir to /root/dreamNav/pairUAV/train for full dataset.