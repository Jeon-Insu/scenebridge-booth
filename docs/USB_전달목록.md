# USB로 전달하는 파일과 넣는 위치

소스 코드는 GitHub(`https://github.com/Jeon-Insu/scenebridge-booth`)에서 받는다. 아래 파일은 용량·저작권·초상권 때문에 저장소에 없어서 USB로 따로 전달한다.

USB의 폴더 구조는 넣을 위치와 똑같이 맞춰 두었다.

- `scenebridge-booth/` 아래 → 받은 저장소 폴더에 **그대로 겹쳐 복사**
- `home/` 아래 → 받는 사람의 **홈 폴더(`~`)에 그대로 복사**

```bash
git clone https://github.com/Jeon-Insu/scenebridge-booth.git
cp -r <USB>/scenebridge-booth/. scenebridge-booth/
cp -r <USB>/home/. ~/
```

## 1. 꼭 필요한 것 (USB 안, 약 1.3GB)

| 무엇 | USB 안 위치 = 넣을 위치 | 크기 | 없으면 |
|---|---|---|---|
| 동신여고 배경 (2026-10-27 최종본) | `scenebridge-booth/usd/venues/source/Dongshin_Final/` (폴더 전체) | 658MB | Isaac 트윈 배경이 안 뜸 (추적·에이전트는 동작) |
| 소품 3D 모델 (Meshy) 12개 | `scenebridge-booth/assets/props/<소품ID>/model.usdz` | 약 150MB | 트윈에서 그 소품이 안 보임 (인식은 동작) |
| 피규어 검출기 A1 | `scenebridge-booth/outputs/models/prop_detector_a1.pth` | 128MB | 추적기가 시작하지 않음 |
| 방향 분류기 | `scenebridge-booth/outputs/models/orientation_head.pt` | 39KB | 추적기가 시작하지 않음 |
| 일반 물체 검출기 (COCO 기본) | `home/.roboflow/models/rf-detr-seg-nano.pt` → `~/.roboflow/models/` | 128MB | 처음 실행 때 인터넷에서 자동으로 받음 (부스에 인터넷이 없으면 필요) |
| DINOv2-small (방향 분류기의 바탕) | `home/.cache/huggingface-gist/hub/models--facebook--dinov2-small/` → `~/.cache/huggingface-gist/hub/` | 85MB | 처음 실행 때 인터넷에서 자동으로 받음 |

소품 3D 모델 12개는 다음과 같다.

| 소품ID(폴더 이름) | 무엇 |
|---|---|
| `giraffe`, `zebra`, `elephant`, `dinosaur`, `devil_duck`, `green_robot` | 피규어 6종 |
| `gist_mascot_grey` | 회색 마스코트 |
| `potted_plant` | 화분 |
| `paper_cup` | 컵 (목록의 "컵"이 이 모델을 씀) |
| `stainless_mug`, `desk_clock`, `gist_mascot` | 예전 목록용 (지금 목록에는 없음, 카탈로그 호환용) |

- 휴대폰·책·페트병 모델은 단순 형상이라 저장소에 이미 들어 있다.
- `model.usdz` 옆의 `model_placed.usda`(저장소에 있음)가 크기·바닥 위치를 맞추므로, **파일 이름과 폴더를 바꾸지 않는다.**

## 2. 시험용 (USB 안, 지금 설정에서는 안 씀)

| 무엇 | 위치 | 설명 |
|---|---|---|
| B안 모델 2차 | `scenebridge-booth/outputs/models/prop_detector_b2.pth` (+ `.classes.json`) | 12종을 모델 하나로 찾는 시험 모델. 실제 녹화에서 로봇 피규어를 거의 못 찾아 **지금은 쓰지 않는다.** 바꾸는 방법은 `docs/HANDOFF.md` 5절 |

B안 3차(피규어 비중을 높인 재학습)는 2026-10-28 오후에 학습 중이었다. 결과가 좋으면 같은 위치로 따로 전달한다.

## 3. USB에 넣지 않은 것 (받는 쪽에서 준비)

| 무엇 | 방법 |
|---|---|
| 언어 모델 Gemma 4 (`nvidia/Gemma-4-26B-A4B-NVFP4`, 약 18GB) | Hugging Face 계정으로 사용 조건에 동의하고 `HF_TOKEN`을 설정하면 `./tools/run_sglang_tabletop.sh start`가 처음 실행 때 받는다. 인터넷이 없으면 이 컴퓨터의 `~/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/`를 같은 위치로 복사 |
| SGLang Docker 이미지 `lmsysorg/sglang:latest-cu130` | `docker pull`로 받음 (`docs/INSTALL.md` 4절) |
| Isaac Sim 6.0.1 | NVIDIA에서 받아 설치 (`docs/INSTALL.md`) |
| 녹화 영상 | 사람이 찍혀 있어 전달하지 않음. 현장에서 새로 녹화 |
| Meshy API 키 | 전달하지 않음. 필요하면 각자 발급해 `.secrets/meshy_api_key`에 둠 |

## 4. 넣은 뒤 확인

```bash
cd scenebridge-booth
ls assets/props/*/model.usdz | wc -l                 # 12
ls outputs/models/                                     # prop_detector_a1.pth, orientation_head.pt (+ b2)
python3 tools/venue_args.py dongshin_ai_classroom      # 인자 여러 줄이 나오면 배경 정상
python3 tools/booth_check.py                           # 소품 목록 점검
```

그다음 `docs/INSTALL.md`(설치)와 `docs/HANDOFF.md`(켜기·운영) 순서로 진행한다.
