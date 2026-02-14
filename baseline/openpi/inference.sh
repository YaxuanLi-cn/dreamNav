
CKPT_DIR="${1:-checkpoints/pi05_ours}"
BATCH_SIZE="${2:-32}"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/inference.py \
  --checkpoint-dir "$CKPT_DIR" \
  --test-dir /root/dreamNav/pairUAV/test \
  --tour-dir /root/dreamNav/pairUAV/tours \
  --batch-size "$BATCH_SIZE"
