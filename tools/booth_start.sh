#!/usr/bin/env bash
# 부스 시작: 소품 목록 확인 → 언어 모델(SGLang) 준비 → 카메라 4대 라이브 데모를 부스 설정으로 켭니다.
#   ./tools/booth_start.sh
# 끄기: 이 터미널에서 Ctrl+C, 또는 Isaac 창을 닫습니다.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
booth_config="smartphone_mocap/config/booth.json"

echo "[1/3] 소품 목록 확인"
if ! python3 tools/booth_check.py; then
  echo "소품 목록(config/booth_props.json)을 고친 뒤 다시 실행하세요." >&2
  exit 1
fi

if ! python3 tools/venue_args.py --config "${booth_config}" >/dev/null; then
  echo "장소 프리셋(config/venues.json, 부스 설정의 venue)을 고친 뒤 다시 실행하세요." >&2
  exit 1
fi

echo "[2/3] 언어 모델 준비 (처음 켤 때는 몇 분 걸립니다)"
if ! curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null 2>&1; then
  ./tools/run_sglang_tabletop.sh start
  for _ in $(seq 1 120); do
    curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "언어 모델이 10분 안에 준비되지 않았습니다. ./tools/run_sglang_tabletop.sh logs 로 확인하세요." >&2
    exit 1
  fi
fi
echo "언어 모델 준비됨"

venue="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("venue","gwangju_acc"))' "${booth_config}")"
echo "[3/3] 데모 시작 (설정: ${booth_config}, 장소: ${venue})"
export TABLETOP_CONFIG="${booth_config}"
unset TABLETOP_VENUE
exec ./tools/run_tabletop_demo.sh "$@"
