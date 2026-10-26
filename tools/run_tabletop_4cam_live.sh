#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
config_path="${TABLETOP_CONFIG:-smartphone_mocap/config/booth.json}"
exec .venv-mocap/bin/python -m smartphone_mocap live \
  --config "${config_path}" "$@"
