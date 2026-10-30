#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Isaac Sim: ISAAC_SIM_DIR, or the newest isaac-sim-standalone-* in the home folder or on the Desktop.
isaac_dir="${ISAAC_SIM_DIR:-$(find "${HOME}" "${HOME}/Desktop" -maxdepth 1 -type d -name 'isaac-sim-standalone-*' \
  -print 2>/dev/null | sort -V | tail -n 1)}"
if [[ -z "${isaac_dir}" || ! -x "${isaac_dir}/isaac-sim.sh" ]]; then
  echo "Isaac Sim을 찾지 못했습니다(${isaac_dir:-홈 폴더와 Desktop에 isaac-sim-standalone-* 없음})." >&2
  echo "설치한 폴더를 알려 주세요: export ISAAC_SIM_DIR=/설치/경로/isaac-sim-standalone-6.0.1-linux-aarch64" >&2
  exit 1
fi
ui_check_args=()
# The venue preset (config/venues.json) gives the background USD and where the table sits in it.
# One source of truth: the tracker config's "venue" (TABLETOP_VENUE overrides, e.g. for a quick look).
config="${TABLETOP_CONFIG:-smartphone_mocap/config/booth.json}"
if [[ -n "${TABLETOP_VENUE:-}" ]]; then
  venue_query=("${TABLETOP_VENUE}")
  venue="${TABLETOP_VENUE}"
else
  [[ "${config}" = /* ]] || config="${project_dir}/${config}"
  venue_query=(--config "${config}")
  venue="${config} 의 venue"
fi
mapfile -t venue_args < <("${project_dir}/.venv-mocap/bin/python" "${project_dir}/tools/venue_args.py" "${venue_query[@]}")
if [[ ${#venue_args[@]} -lt 4 ]]; then
  echo "장소 프리셋 '${venue}'을 읽지 못했습니다 (config/venues.json)." >&2
  exit 1
fi
if [[ "${TABLETOP_UI_CHECK:-0}" == "1" ]]; then
  ui_check_args=(--exec "${project_dir}/tools/validate_tabletop_ui.py")
fi

exec "${isaac_dir}/isaac-sim.sh" \
  --ext-folder "${project_dir}/exts" \
  --enable omni.kit.language.core \
  --enable omni.kit.language.korean \
  --enable gist.tabletop_twin \
  --/persistent/app/locale_id=ko-KR \
  --/isaac/startup/create_new_stage=false \
  "${venue_args[@]}" \
  "${ui_check_args[@]}" "$@"
