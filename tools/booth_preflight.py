"""부스를 켜기 전 점검: Python 환경, 모델 가중치, 카메라 장치, Docker 권한 (표준 라이브러리만 사용).

  python3 tools/booth_preflight.py [--config smartphone_mocap/config/booth.json]

문제가 있으면 무엇을 어디에 두거나 고쳐야 하는지 한국어로 알려 주고 1로 끝난다.
경고(예: 일반 물체 모델 B3 없음)는 알리기만 하고 계속한다. tools/booth_start.sh가 먼저 실행한다.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(config_path: Path, project_dir: Path = ROOT) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    config_path = Path(config_path)
    base = config_path.parent
    if not (project_dir / ".venv-mocap/bin/python").exists():
        errors.append("Python 환경(.venv-mocap)이 없습니다. docs/INSTALL.md 1절대로 만드세요.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    od = config.get("object_detection", {})

    def model(key, *, required):
        if not od.get(key):
            return
        path = (base / od[key]).resolve()
        if path.is_file():
            return
        rel = path.relative_to(project_dir) if path.is_relative_to(project_dir) else path
        if required:
            errors.append(f"모델 가중치가 없습니다: {rel} (USB로 받은 파일을 이 위치에 넣으세요, docs/USB_전달목록.md)")
        else:
            warnings.append(f"일반 물체 모델이 없습니다: {rel}. COCO 기본 모델로 대신 찾습니다(회색 마스코트는 거의 못 찾음).")

    model("checkpoint", required=True)
    model("orientation_head", required=True)
    model("general_checkpoint", required=False)

    missing = [c for c in config.get("cameras", [])
               if isinstance(c.get("source"), str) and c["source"].startswith("/") and not Path(c["source"]).exists()]
    for camera in missing:
        errors.append(f"카메라 {camera.get('id')} 장치가 없습니다: {camera['source']}")
    if missing:
        errors.append("카메라 연결을 확인하고, 이 컴퓨터의 장치 경로를 `ls -l /dev/v4l/by-path/`로 본 뒤 "
                      f"{config_path.name}의 cameras[].source를 고치세요(docs/INSTALL.md 5절).")
    return errors, warnings


def docker_warning() -> str | None:
    if shutil.which("docker") is None:
        return "docker가 없습니다. 언어 모델(SGLang)을 켜려면 docs/INSTALL.md 4절대로 설치하세요."
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0 and "permission denied" in (result.stderr or "").lower():
        return "docker 권한이 없습니다: sudo usermod -aG docker $USER 후 다시 로그인하세요."
    return None


GEMMA_CACHE = ".cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4"


def llm_warning(home: Path, env) -> str | None:
    """SGLang downloads Gemma 4 (~18GB) on first start; without a cached copy it needs a Hugging Face token."""
    if (Path(home) / GEMMA_CACHE).is_dir() or env.get("HF_TOKEN"):
        return None
    return ("언어 모델(Gemma 4, 약 18GB)이 아직 없고 HF_TOKEN도 없습니다. 모델 사용 조건에 동의한 계정의 토큰으로 "
            "export HF_TOKEN=... 후 다시 실행하세요(docs/INSTALL.md 4절). 언어 모델 없이도 인식·트윈은 동작합니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "smartphone_mocap/config/booth.json")
    args = parser.parse_args()
    errors, warnings = check(args.config.resolve())
    llm = llm_warning(Path.home(), os.environ)
    if llm:
        warnings.append(llm)
    docker = docker_warning()
    if docker:
        errors.append(docker)
    for line in warnings:
        print("[주의]", line)
    for line in errors:
        print("[문제]", line, file=sys.stderr)
    if not errors:
        print("켜기 전 점검 통과")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
