#!/bin/bash

SESSION_NAME="sft_model"

# Get directory where this script is stored
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set work directory as the parent of script directory
WORKDIR="$(dirname "$SCRIPT_DIR")"

# Tmux binary
TMUX_BIN="tmux"

# Check if session exists
if "$TMUX_BIN" has-session -t $SESSION_NAME 2>/dev/null; then
  echo "Session '$SESSION_NAME' found. Terminating..."
  "$TMUX_BIN" kill-session -t $SESSION_NAME
  echo "Success: Session '$SESSION_NAME' has been killed."
  exit 0
else
  echo "No active session named '$SESSION_NAME' found. Nothing to do."
  exit 0
fi