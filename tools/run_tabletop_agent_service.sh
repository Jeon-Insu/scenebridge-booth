#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
exec .venv-mocap/bin/python -m smartphone_mocap.agent_service \
  --config smartphone_mocap/config/tabletop_agent.json
