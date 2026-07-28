#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${LEROBOT_PYTHON:-/home/yanrihong/miniconda3/envs/lerobot/bin/python}"
SESSION_NAME="${FRANKA_RECORD_UI_SESSION:-lerobot_franka_record_ui}"

usage() {
    echo "Usage: $0 {start-ui|stop-ui|status|help} [recorder arguments]"
    echo "  start-ui  Start the Tk recorder UI in one tmux session"
    echo "  stop-ui   Request graceful UI shutdown (never force-kills the robot process)"
    echo "  status    Show whether the recorder UI session is running"
}

require_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is required to manage the recorder UI" >&2
        exit 1
    fi
}

session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

quote_command() {
    local -a command=(
        env
        "PYTHONPATH=$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    )
    if [[ -n "${DISPLAY:-}" ]]; then
        command+=("DISPLAY=$DISPLAY")
    fi
    if [[ -n "${XAUTHORITY:-}" ]]; then
        command+=("XAUTHORITY=$XAUTHORITY")
    fi
    command+=("$PYTHON_BIN" "$REPO_ROOT/hardware_test/franka/run_record_ui.py")
    command+=("$@")
    printf '%q ' "${command[@]}"
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    start-ui)
        require_tmux
        if [[ ! -x "$PYTHON_BIN" ]]; then
            echo "LeRobot Python is not executable: $PYTHON_BIN" >&2
            exit 1
        fi
        if session_exists; then
            echo "Recorder UI is already running in tmux session: $SESSION_NAME" >&2
            exit 1
        fi
        ui_command="$(quote_command "$@")"
        tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$ui_command"
        echo "Started Franka recorder UI: tmux session $SESSION_NAME"
        ;;
    stop-ui)
        require_tmux
        if ! session_exists; then
            echo "Recorder UI is not running"
            exit 0
        fi
        tmux send-keys -t "$SESSION_NAME" C-c
        for _ in $(seq 1 300); do
            if ! session_exists; then
                echo "Recorder UI stopped cleanly"
                exit 0
            fi
            sleep 0.1
        done
        echo "Recorder UI is still waiting for an on-screen save/discard decision" >&2
        exit 1
        ;;
    status)
        require_tmux
        if session_exists; then
            echo "Recorder UI is running: tmux session $SESSION_NAME"
            exit 0
        fi
        echo "Recorder UI is not running"
        exit 1
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
