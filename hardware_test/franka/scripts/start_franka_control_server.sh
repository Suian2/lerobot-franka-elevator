#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CONTROL_HOST="$({
  PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c \
    'from hardware_test.franka.defaults import get_control_host; print(get_control_host())'
})"
REMOTE="${FRANKA_CONTROL_REMOTE:-franka@${CONTROL_HOST}}"
CONTROL_DOCKER_DIR="${FRANKA_CONTROL_DOCKER_DIR:-/home/franka/franka_ws/base/teleop/docker}"
CONTROL_SESSION="${FRANKA_CONTROL_SESSION:-vita-franka-server}"
CONTROL_LOG="${FRANKA_CONTROL_LOG:-/tmp/${CONTROL_SESSION}.log}"

IMAGE="${IMAGE:-vita-franka-server:zmq-franky-tuned}"
CONTAINER_NAME="${CONTAINER_NAME:-vita-franka-server}"

# Physically tested VITA fallback parameters from 2026-04-27.
FRANKY_VELOCITY="${FRANKA_FRANKY_DYNAMICS_VELOCITY:-0.15}"
FRANKY_ACCELERATION="${FRANKA_FRANKY_DYNAMICS_ACCELERATION:-0.10}"
FRANKY_JERK="${FRANKA_FRANKY_DYNAMICS_JERK:-0.10}"

SERVER_URL="${FRANKA_SERVER_URL:-http://${CONTROL_HOST}:29000/ctl}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  start-control   Replace and start the Docker franka_server on the control machine
  status          Show remote Docker/tmux and HTTP/ZMQ health
  stop-control    Stop the remote Docker server and tmux session

The default control host comes from hardware_test/franka/defaults.py.

Useful overrides:
  FRANKA_CONTROL_HOST=${CONTROL_HOST}
  FRANKA_CONTROL_REMOTE=${REMOTE}
  FRANKA_CONTROL_PASSWORD='...'  # omit when SSH keys work
  FRANKA_CONTROL_DOCKER_DIR=${CONTROL_DOCKER_DIR}
  FRANKA_CONTROL_SESSION=${CONTROL_SESSION}
  IMAGE=${IMAGE}
  CONTAINER_NAME=${CONTAINER_NAME}
  FRANKA_FRANKY_DYNAMICS_VELOCITY=${FRANKY_VELOCITY}
  FRANKA_FRANKY_DYNAMICS_ACCELERATION=${FRANKY_ACCELERATION}
  FRANKA_FRANKY_DYNAMICS_JERK=${FRANKY_JERK}
EOF
}

require_local_commands() {
  local command_name
  for command_name in ssh curl "${PYTHON_BIN}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "Required local command not found: ${command_name}" >&2
      exit 1
    fi
  done
  if [[ -n "${FRANKA_CONTROL_PASSWORD:-}" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "FRANKA_CONTROL_PASSWORD is set, but sshpass is not installed." >&2
    exit 1
  fi
}

remote_shell() {
  local -a ssh_command=(ssh)
  if [[ -n "${FRANKA_CONTROL_PASSWORD:-}" ]]; then
    ssh_command=(sshpass -p "${FRANKA_CONTROL_PASSWORD}" ssh)
  fi
  "${ssh_command[@]}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ProxyCommand=none \
    "${REMOTE}" "$@"
}

remote_shell_quiet() {
  remote_shell "$@" 2>/dev/null || true
}

wait_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-45}"
  local _
  for _ in $(seq 1 "${attempts}"); do
    if curl --noproxy '*' -fsS -m 1 "${url}" >/dev/null 2>&1; then
      echo "${label}: ready"
      return 0
    fi
    sleep 1
  done
  echo "${label}: not ready after ${attempts}s" >&2
  return 1
}

start_control() {
  echo "Starting control server on ${REMOTE} with image ${IMAGE}"
  remote_shell "mkdir -p '${CONTROL_DOCKER_DIR}' && rm -f '${CONTROL_LOG}'"
  remote_shell "docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true; tmux kill-session -t '${CONTROL_SESSION}' >/dev/null 2>&1 || true"
  remote_shell "cd '${CONTROL_DOCKER_DIR}' && tmux new-session -d -s '${CONTROL_SESSION}' 'IMAGE=${IMAGE} CONTAINER_NAME=${CONTAINER_NAME} FRANKA_VELOCITY_BACKEND=franky FRANKA_REALTIME_IGNORE=1 FRANKA_FRANKY_DYNAMICS_VELOCITY=${FRANKY_VELOCITY} FRANKA_FRANKY_DYNAMICS_ACCELERATION=${FRANKY_ACCELERATION} FRANKA_FRANKY_DYNAMICS_JERK=${FRANKY_JERK} FRANKA_FRANKY_STOP_DYNAMICS_VELOCITY=${FRANKY_VELOCITY} FRANKA_FRANKY_STOP_DYNAMICS_ACCELERATION=${FRANKY_ACCELERATION} FRANKA_FRANKY_STOP_DYNAMICS_JERK=${FRANKY_JERK} ./run_franka_server_docker_control_machine.sh 2>&1 | tee -a ${CONTROL_LOG}'"
  echo "Waiting for control server health at ${SERVER_URL}/config"
  if ! wait_http "${SERVER_URL}/config" "control server"; then
    echo
    echo "== Remote control server log (${CONTROL_LOG}) =="
    remote_shell_quiet "test -f '${CONTROL_LOG}' && tail -n 160 '${CONTROL_LOG}' || echo 'No remote control log found.'"
    return 1
  fi
  curl --noproxy '*' -fsS "${SERVER_URL}/config"
  echo
}

status() {
  echo "== Remote control server on ${REMOTE} =="
  remote_shell_quiet "docker ps --filter name='${CONTAINER_NAME}' --format '{{.Names}} {{.Image}} {{.Status}}'; tmux has-session -t '${CONTROL_SESSION}' 2>/dev/null && echo '${CONTROL_SESSION}: tmux running' || echo '${CONTROL_SESSION}: tmux stopped'"
  echo
  echo "== Server health =="
  curl --noproxy '*' -m 2 -sS "${SERVER_URL}/config" || true
  echo
  curl --noproxy '*' -m 2 -sS "${SERVER_URL}/velocity_zmq_status" || true
  echo
  curl --noproxy '*' -m 2 -sS "${SERVER_URL}/velocity_ws_status" || true
  echo
}

stop_control() {
  echo "Stopping control server on ${REMOTE}"
  remote_shell "docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true; tmux kill-session -t '${CONTROL_SESSION}' >/dev/null 2>&1 || true"
}

require_local_commands

case "${1:-}" in
  start-control) start_control ;;
  status) status ;;
  stop-control) stop_control ;;
  -h|--help|help|"") usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
