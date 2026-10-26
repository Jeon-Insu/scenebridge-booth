#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if python3 -c 'from pxr import Usd' >/dev/null 2>&1; then
  exec python3 "$project_root/tools/normalize_prop_assets.py" "$@"
fi

isaac_root="${ISAAC_SIM_ROOT:-}"
if [[ -z "$isaac_root" ]]; then
  isaac_root="$(find "${HOME}" "${HOME}/Desktop" -maxdepth 1 -type d -name 'isaac-sim-standalone-*' -print 2>/dev/null | sort -V | tail -n 1)"
fi
if [[ -z "$isaac_root" || ! -d "$isaac_root/extscache" ]]; then
  echo "Pixar USD Python module not found. Set ISAAC_SIM_ROOT and retry." >&2
  exit 1
fi

usd_lib_dir="$(find "$isaac_root/extscache" -maxdepth 1 -type d -name 'omni.usd.libs-*' -print | sort -V | tail -n 1)"
if [[ -z "$usd_lib_dir" ]]; then
  echo "Isaac Sim USD libraries were not found under $isaac_root/extscache." >&2
  exit 1
fi

export PYTHONPATH="$usd_lib_dir${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$usd_lib_dir/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PXR_PLUGINPATH_NAME="$usd_lib_dir/bin/usd${PXR_PLUGINPATH_NAME:+:$PXR_PLUGINPATH_NAME}"
exec python3 "$project_root/tools/normalize_prop_assets.py" "$@"
