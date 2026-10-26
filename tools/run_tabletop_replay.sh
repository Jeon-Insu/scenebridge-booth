#!/usr/bin/env bash
# Input-only recovery: keep the existing Isaac and agent processes.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 || ! -f "$1/session.json" ]]; then
  echo "Usage: $0 /path/to/recorded-session (session.json required)" >&2
  exit 2
fi
session_dir="$(cd "$1" && pwd)"
cd "${project_dir}"
replay_pid=""
stop_replay() {
  trap - INT TERM EXIT
  if [[ -n "${replay_pid}" ]]; then
    kill -TERM "${replay_pid}" 2>/dev/null || true
    wait "${replay_pid}" 2>/dev/null || true
  fi
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap stop_replay EXIT
echo "Replay source: ${session_dir} | start: ${TABLETOP_REPLAY_START:-0}"
while true; do
  .venv-mocap/bin/python -m smartphone_mocap replay \
    --session "${session_dir}" \
    --output-dir "${project_dir}/outputs/recorded_demo_replay" \
    --start "${TABLETOP_REPLAY_START:-0}" \
    --publish-udp --embedded-preview --async-perception --realtime --speed 1.0 &
  replay_pid="$!"
  wait "${replay_pid}"
  replay_pid=""
done
