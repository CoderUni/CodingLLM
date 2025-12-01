#!/bin/bash

# WIP (WORK IN PROGRESS)

# Configuration
SESSION_NAME="model_eval"
WORKDIR="/mnt/storage/metnet/coding_llm"
VENV_PATH="LiveCodeBench/livecodebench/bin/activate" 

# Huggingface model name
MODEL_NAME="BigJuicyData/Anni"
RELEASE_VERSION="release_v6"
MAX_TOKENS=28000
TEMPERATURE=0.2
TOP_P=0.95
START_DATE="2025-03-01"
END_DATE="2025-12-01"
N_SAMPLES=1 # Number of samples per prompt (We only test PASS@1)

TMUX_BIN="/home/metnet/miniconda3/bin/tmux"

mkdir -p "$LOG_DIR"

# Check if session exists
if "$TMUX_BIN" has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attaching..."
  "$TMUX_BIN" attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session '$SESSION_NAME'..."

# Run this inside cloned LiveCodeBench folder
"$TMUX_BIN" new-session -d -s $SESSION_NAME "bash --norc -c '
  cd $WORKDIR &&
    echo \"Activating Venv: $VENV_PATH\";
  if [ -f \"$VENV_PATH\" ]; then
      source \"$VENV_PATH\"
  else
      echo \"ERROR: Virtual environment not found at $VENV_PATH\"
      echo \"Please edit the script to point to your python environment.\"
      read -p \"Press enter to exit...\"
      exit 1
  fi

  echo \"Starting vLLM server..\" &&

  cd LiveCodeBench &&

  export CUDA_VISIBLE_DEVICES=1 &&
  
  python -m lcb_runner.runner.main \
      --model "$MODEL_NAME" \
      --scenario codegeneration \
      --release_version "$RELEASE_VERSION" \
      --start_date "$START_DATE" \
      --end_date "$END_DATE" \
      --max_tokens $MAX_TOKENS \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --n $N_SAMPLES \
      --evaluate &&

  echo \"Server stopped. Press Ctrl+C to close.\";
  exec bash
'"

#  git clone https://github.com/LiveCodeBench/LiveCodeBench.git &&
#  pip install -e . &&

echo "LiveCodeBench eval started."
echo "API URL: http://localhost:8000/v1"
echo "Attach with: $TMUX_BIN attach -t $SESSION_NAME"