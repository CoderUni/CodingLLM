#!/bin/bash

# Configuration
SESSION_NAME="serve_model"
WORKDIR="/mnt/storage/metnet/coding_llm"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/serve_$(date +'%Y%m%d_%H%M%S').log"
MODEL_PATH="../models/final_finetuned_model"
MODEL_NAME="BigJuicyData/Anni"

TMUX_BIN="/home/metnet/miniconda3/bin/tmux"

mkdir -p "$LOG_DIR"

# Check if session exists
if "$TMUX_BIN" has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attaching..."
  "$TMUX_BIN" attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session '$SESSION_NAME'..."

"$TMUX_BIN" new-session -d -s $SESSION_NAME "bash --norc -c '
  cd $WORKDIR &&
  source /home/metnet/venvs/unsloth_env/bin/activate &&
  echo \"Starting vLLM server for $MODEL_PATH...\" &&
  vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 32000 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 | tee $LOG_FILE;
  echo \"Server stopped. Press Ctrl+C to close.\";
  exec bash
'"

echo "Server started. Logs: $LOG_FILE"
echo "API URL: http://localhost:8000/v1"
echo "Attach with: $TMUX_BIN attach -t $SESSION_NAME"