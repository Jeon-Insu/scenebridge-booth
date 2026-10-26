#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${project_dir}/outputs/logs"
mkdir -p "${log_dir}"

isaac_pid=""
tracker_pid=""
agent_pid=""

stop_children() {
  trap - INT TERM EXIT
  if [[ -n "${tracker_pid}" ]] && kill -0 "${tracker_pid}" 2>/dev/null; then
    kill -INT "${tracker_pid}" 2>/dev/null || true
  fi
  if [[ -n "${agent_pid}" ]] && kill -0 "${agent_pid}" 2>/dev/null; then
    kill -INT "${agent_pid}" 2>/dev/null || true
  fi
  if [[ -n "${isaac_pid}" ]] && kill -0 "${isaac_pid}" 2>/dev/null; then
    kill -INT "${isaac_pid}" 2>/dev/null || true
  fi
  [[ -z "${tracker_pid}" ]] || wait "${tracker_pid}" 2>/dev/null || true
  [[ -z "${agent_pid}" ]] || wait "${agent_pid}" 2>/dev/null || true
  [[ -z "${isaac_pid}" ]] || wait "${isaac_pid}" 2>/dev/null || true
}
trap stop_children INT TERM EXIT

"${project_dir}/tools/run_tabletop_agent_service.sh" \
  >"${log_dir}/tabletop_agent.log" 2>&1 &
agent_pid="$!"
echo "Local scene-agent bridge started (PID ${agent_pid}); SGLang is used when available."

"${project_dir}/tools/run_isaac_tabletop_perception.sh" \
  >"${log_dir}/tabletop_isaac.log" 2>&1 &
isaac_pid="$!"

echo "Isaac Sim starting (PID ${isaac_pid}); log: ${log_dir}/tabletop_isaac.log"
isaac_ready=false
for _ in $(seq 1 60); do
  kill -0 "${isaac_pid}" 2>/dev/null || {
    echo "Isaac Sim exited during startup. See ${log_dir}/tabletop_isaac.log" >&2
    exit 1
  }
  if rg -q "app ready|Isaac Sim Full App is loaded" "${log_dir}/tabletop_isaac.log" 2>/dev/null; then
    isaac_ready=true
    break
  fi
  sleep 0.5
done
if [[ "${isaac_ready}" != true ]]; then
  echo "Isaac Sim did not report ready within 30 seconds." >&2
  exit 1
fi

"${project_dir}/tools/run_tabletop_4cam_live.sh" "$@" \
  >"${log_dir}/tabletop_tracker.log" 2>&1 &
tracker_pid="$!"
echo "Object tracker started (PID ${tracker_pid}); log: ${log_dir}/tabletop_tracker.log"
echo "Layout: large Viewport, 8-view camera comparison in Stage, Settings/Agent below."
echo "Close Isaac Sim or press Ctrl+C here to stop both processes."

set +e
wait -n "${isaac_pid}" "${tracker_pid}" "${agent_pid}"
status="$?"
set -e
exit "${status}"
