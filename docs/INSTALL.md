# 설치

## 검증한 환경

아래 환경에서 개발하고 확인했다. 다른 환경은 아직 검증하지 않았다.

| 항목 | 값 |
|---|---|
| 컴퓨터 | NVIDIA GB10 (aarch64 ARM, CPU·GPU 통합 메모리) |
| OS | Ubuntu 24.04.4 LTS, NVIDIA 드라이버 580.142 |
| Python | 3.12.3 (venv `.venv-mocap`) |
| Isaac Sim | 6.0.1 standalone (linux-aarch64) |
| 언어 모델 | Docker + NVIDIA Container Toolkit, SGLang `lmsysorg/sglang:latest-cu130`, `nvidia/Gemma-4-26B-A4B-NVFP4` |
| 카메라 | Creative Live! Cam ×4 (USB UVC, 1280×720) |

일반 x86 + 단일 GPU 컴퓨터에서는 세 가지가 달라진다.
- Isaac Sim은 linux-x86_64판을 받고 `ISAAC_SIM_DIR`로 위치를 알려 준다.
- PyTorch는 그 GPU에 맞는 CUDA 빌드를 설치한다.
- SGLang 메모리 설정을 줄여야 한다(4절). GB10의 64K context·메모리 비율 0.40 설정은 통합 메모리 기준이라 일반 GPU VRAM에 그대로 맞지 않는다.

## 1. 저장소와 Python 환경

```bash
git clone https://github.com/Jeon-Insu/scenebridge-booth.git
cd scenebridge-booth
python3.12 -m venv .venv-mocap
# PyTorch를 먼저 설치한다. 플랫폼에 맞는 CUDA 빌드여야 한다(검증: torch 2.13.0, torchvision 0.28.0).
.venv-mocap/bin/python -m pip install torch torchvision --index-url <플랫폼에 맞는 PyTorch 주소>
.venv-mocap/bin/python -m pip install -r smartphone_mocap/requirements.txt
.venv-mocap/bin/python -m pytest tests -q          # 회귀 테스트 (2026-10-26: 419 passed)
```

- 방향 분류기가 쓰는 DINOv2-small(`facebook/dinov2-small`)은 처음 실행할 때 Hugging Face에서 자동으로 받는다. 받는 위치는 `~/.cache/huggingface-gist/hub`이고, 환경변수 `HF_HUB_CACHE`로 바꿀 수 있다.
- 재학습까지 하려면 학습용 의존성을 추가로 설치한다.
  ```bash
  .venv-mocap/bin/python -m pip install --no-deps -r smartphone_mocap/requirements-train.txt
  ```
  `--no-deps`를 붙이는 이유: 붙이지 않으면 화면 없는 OpenCV(headless)가 함께 깔려, 미리보기 창에 쓰는 OpenCV와 충돌한다.

## 2. 모델 가중치 (따로 전달)

모델 가중치 두 파일은 저장소에 없고 따로 전달받는다. 받은 파일을 이름 그대로 `outputs/models/`에 넣는다.

```
outputs/models/prop_detector_a1.pth
outputs/models/orientation_head.pt
```

| 파일 | 내용 |
|---|---|
| `prop_detector_a1.pth` (128MB) | RF-DETR-Seg nano, 소품 10종, Isaac 합성 데이터로 학습 |

일반 물체(컵·휴대폰·책·화분·페트병·회색 마스코트)를 찾는 COCO 80종 모델(`rf-detr-seg-nano.pt`)은 **처음 실행할 때 인터넷에서 자동으로 받아** `~/.roboflow/models/`에 저장된다. 부스에 인터넷이 없으면 미리 한 번 실행해 두거나 이 파일을 같은 위치에 복사한다.
| `orientation_head.pt` | DINOv2-small 특징 위에 붙인 선형 방향 분류기 |

## 3. 따로 전달받는 파일

배경 USD와 소품 Meshy 모델을 넣는다. 위치와 파일 이름 규칙은 [`docs/ASSETS.md`](ASSETS.md)에 있다.

## 4. 언어 모델 (SGLang + Gemma 4)

```bash
export HF_TOKEN=<Hugging Face 토큰>      # 모델 사용 조건에 동의한 계정
./tools/run_sglang_tabletop.sh start    # 처음에는 모델 내려받기로 시간이 걸린다
./tools/run_sglang_tabletop.sh logs     # 상태 보기
curl http://127.0.0.1:30000/v1/models   # 모델 이름이 나오면 준비 완료
```

- 기본 프로필은 `gemma4-26b-a4b`(GB10용)이다. 다른 GPU에서는 아래 환경변수로 메모리를 맞춘다.
  - `SGLANG_MEMORY_FRACTION`
  - `SGLANG_CONTEXT_LENGTH`
  - `SGLANG_MAX_TOTAL_TOKENS`
- 작은 모델 프로필 `SGLANG_PROFILE=qwen3-8b`도 스크립트에 있다. 다만 이 데모에서는 Gemma 4만 검증했다.
- 에이전트 서비스(HTTP 7080)는 데모 시작 스크립트가 함께 켠다. 설정 파일은 `smartphone_mocap/config/tabletop_agent.json`이다.

## 5. 카메라

1. 카메라 4대를 연결하고 장치 경로를 확인한다.
   ```bash
   ls -l /dev/v4l/by-path/
   ```
2. `smartphone_mocap/config/booth.json`의 `cameras[].source`를 자기 장치 경로로 바꾼다. 형식(`fourcc`, `fps`)은 대역폭에 맞춘다.
   - 허브 하나를 두 대가 같이 쓰면 YUYV 10fps로 둔다.
   - 따로 연결된 카메라는 MJPG 15fps로 둔다.
3. 허브 하나를 두 대가 같이 써서 한 대가 0fps가 되면 USB 대역폭 보정을 적용한다.
   ```bash
   sudo bash tools/creative_uvc_bandwidth.sh apply
   ```
   재부팅하거나 그 카메라를 다시 꽂을 때마다 다시 적용해야 한다.
4. 카메라 위치가 저장소의 보정(`smartphone_mocap/calibration/venue_20261025`, 2026-10-25 실험실 설치)과 다르면 **외부 보정을 새로 한다.** 순서는 [`docs/HANDOFF.md`](HANDOFF.md)의 보정 절에 있다. 보정은 카메라 위치에 묶여 있어서, 다른 장소에서는 반드시 다시 해야 한다.

## 6. 확인과 시작

```bash
python3 tools/booth_check.py                          # 소품 목록 점검
python3 tools/venue_args.py dongshin_ai_classroom      # 배경 파일 확인
.venv-mocap/bin/python tools/booth_smoke_test.py      # 에이전트 점검 (SGLang·에이전트 서비스가 켜져 있어야 함)
./tools/booth_start.sh                                # 목록 확인 → 언어 모델 → 데모
```

Isaac Sim 위치가 `~/isaac-sim-standalone-6.0.1-linux-aarch64`가 아니면 먼저 지정한다.

```bash
export ISAAC_SIM_DIR=<Isaac Sim 폴더>
```
