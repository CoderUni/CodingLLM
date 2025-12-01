#!/bin/bash


# Set work directory relative to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKDIR="$(dirname "$SCRIPT_DIR")"

# ==========================================
# USER CONFIGURATION
# ==========================================

# Change this!
VENV_PATH="/home/metnet/venvs/unsloth_env/bin/activate" 

# Tmux session name
SESSION_NAME="sft_model"

# Log Handling
LOG_DIR="$WORKDIR/logs"
LOG_FILE="$LOG_DIR/train_$(date +'%Y%m%d_%H%M%S').log"

# Tmux Binary
# Use system default 'tmux' for portability.
# Use absolute path if you plan to use cron.
TMUX_BIN="tmux"
# TMUX_BIN="/home/metnet/miniconda3/bin/tmux" 

# ==========================================
# EXECUTION
# ==========================================

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$WORKDIR"

# Check if tmux is installed
if ! command -v "$TMUX_BIN" &> /dev/null; then
    echo "Error: tmux command not found at '$TMUX_BIN'"
    exit 1
fi

# Check if session exists
if "$TMUX_BIN" has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attaching..."
  "$TMUX_BIN" attach -t $SESSION_NAME
  exit 0
fi

echo "Starting new tmux session '$SESSION_NAME'..."

"$TMUX_BIN" new-session -d -s $SESSION_NAME "bash --norc -c '
  echo \"Changing directory to: $WORKDIR\";
  cd \"$WORKDIR\" || { echo \"Failed to cd to $WORKDIR\"; exec bash; }
  
  echo \"Activating Venv: $VENV_PATH\";
  if [ -f \"$VENV_PATH\" ]; then
      source \"$VENV_PATH\"
  else
      echo \"ERROR: Virtual environment not found at $VENV_PATH\"
      echo \"Please edit the script to point to your python environment.\"
      read -p \"Press enter to exit...\"
      exit 1
  fi

  # Run the training script
  echo \"Starting Python script...\"
  python -u -m src.train | tee \"$LOG_FILE\";
  
  # Alternate launch command (commented out)
  # accelerate launch -m src.train --num_processes=2 --num_machines=1 --mixed_precision=bf16 | tee \"$LOG_FILE\";

  echo \"Training finished. Press Ctrl+C to close.\";
  exec bash
'"

echo "Training started."
echo "Logs: $LOG_FILE"
echo "Attach with: $TMUX_BIN attach -t $SESSION_NAME"