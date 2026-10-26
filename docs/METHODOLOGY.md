# 방법론

SceneBridge 부스(v1.2)가 카메라 영상을 디지털 트윈으로 옮기는 방법을 정리한다. 다루는 순서는 다음과 같다.

1. 전체 흐름
2. 인식 모델 학습
3. 3D 위치와 추적
4. 방향
5. 카메라 보정
6. 트윈과 장소
7. 에이전트와 물리 실험
8. 평가 결과
9. 한계

수치는 출처를 함께 적는다. **합성 데이터 수치는 현장 정확도가 아니다.**

## 1. 전체 흐름

```
USB 카메라 4대 (또는 녹화 AVI 4개)
 → 촬영 시각으로 묶기 (허용 100ms, 2대 이상이면 처리)
 → 찾기·이름: RF-DETR-Seg nano "A1" (클래스 = 소품 ID 10종, 부스 목록에 있는 것만 허용)
 → 묶기·3D: 카메라별 로컬 추적 → 시점 간 연결(object-centric) → 마스크 대표점 삼각측량 → 테이블 평면 지지
 → 합치기·잔상: 같은 소품 3cm 안 분할은 합침, 옮긴 직후 15cm 안 잔상은 숨김
 → 개수 전제: 목록의 count만큼만 (카메라 수 → 신뢰도 순)
 → 추적: ID 유지, 1.5초 동안 안 보이면 사라짐, 다시 나타나면 같은 ID
 → 방향: 0.5초마다 DINOv2-small + 선형층, 카메라 간 원형 평균
 → UDP 7001 JSON → Isaac Sim 확장 gist.tabletop_twin → 소품 USD 배치
에이전트: Isaac 대화창 → HTTP 7080 → SGLang(Gemma 4 26B-A4B NVFP4) → 도구 → Isaac PhysX
```

- 코드 위치:

  | 단계 | 파일 |
  |---|---|
  | 전체 처리 | `smartphone_mocap/perception.py` |
  | 개수 전제, 방향 | `recognition/objects.py`, `recognition/orientation.py` |
  | 묶기·3D·추적 | `object_centric_association.py`, `object_reconstruction.py` |
  | 메시지 | `publisher.py` |
  | 트윈 | `exts/gist.tabletop_twin` |

- 좌표계(테이블 프레임):
  - 원점은 카메라 B 쪽 앞 왼쪽 모서리이고, 테이블은 1.25×0.75m다.
  - +X는 B에서 D 방향, +Y는 벽 쪽(조작자 반대편), +Z는 위다.
  - `xyz`는 물체 바닥 중심(m)이다.

## 2. 인식 모델: 실물 사진 없이 합성 데이터로 학습

**왜 합성인가.** 부스 소품은 장난감 피규어라 공개 데이터셋에 없다. 일반 검출기(COCO 80종)는 로봇·오리·공룡을 찾지 못하거나 다른 이름을 붙였다. 그래서 소품마다 Meshy로 3D 모델을 만들고, Isaac Replicator로 실제 카메라 배치를 흉내 낸 장면을 렌더해 학습했다.

**장면 무작위화** (`smartphone_mocap/synthdata/scene_sampler.py`)
- 소품 배치와 자세: 서 있음 0.5, 누움 0.4(앞·뒤·좌·우), 뒤집힘 0.1.
- 카메라: 현장 보정 배치 30%와 테이블을 겨누는 무작위 배치 70%를 섞는다(±15cm, ±10°).
- 테이블 크기, 조명, CC0 질감도 무작위로 바꾼다.
- 선택: NVIDIA 자산을 라벨 없는 방해물로 넣는다.
- 현장 열화: 과노출, 번짐, 플레어, JPEG, 흐림(`synthdata/degrade.py`).
- 정답:
  - 마스크는 렌더러의 물체 ID 이미지에서 만든다.
  - 위치는 렌더러가 기록한 실제 경계상자 중심(`placed.json`)을 쓴다. 모델 원점은 누운 자세에서 틀리기 때문이다.

**만드는 순서**

```bash
.venv-mocap/bin/python -m smartphone_mocap.synthdata.textures --out assets/textures      # CC0 질감, 1회
.venv-mocap/bin/python -m smartphone_mocap.synthdata.scene_sampler \
    --calibration-dir smartphone_mocap/calibration/venue_20261025 --count 400 \
    --class-unit asset --camera-layout mixed --output outputs/synthetic/specs.json
$ISAAC_SIM_DIR/python.sh tools/generate_synthetic_props.py \
    --specs outputs/synthetic/specs.json --raw-dir outputs/synthetic/raw [--skip-done]      # 데모 Isaac을 끈 상태에서
.venv-mocap/bin/python -m smartphone_mocap.synthdata.instances --raw outputs/synthetic/raw --specs outputs/synthetic/specs.json
.venv-mocap/bin/python -m smartphone_mocap.synthdata.coco_dataset package --raw outputs/synthetic/raw --out datasets/props_synth_<날짜>
.venv-mocap/bin/python tools/train_props.py --dataset datasets/props_synth_<날짜> --output models/props_<날짜> --epochs 10
```

- 렌더는 40장면에 약 2분 걸린다. 수십~수백 장면마다 Isaac이 `rtx.syntheticdata` 오류로 멈출 수 있다. 그때는 `--skip-done`으로 이어서 렌더한다.
- **A1**: 400장면, 10 epoch, 해상도 312, 합성 검증 mAP 0.547이다. 학습한 체크포인트(`checkpoint_best_total.pth`)를 `outputs/models/`에 두고 `booth.json`의 `object_detection.checkpoint`로 가리킨다.
- **고친 버그 두 가지**:
  - 렌더 카메라의 영상 중심 이동 부호가 반대여서, 투영이 (−20, +72)px 어긋났다. 고친 뒤 차이는 1~4px다.
  - 정규화 축척이 덮어써지는 버그가 있었다. 이 버그가 있던 때의 합성 데이터는 버렸다.

## 3. 3D 위치와 추적

- **시점 간 연결**: 카메라마다 로컬 추적 ID를 두고, 같은 종류이면서 재투영 오차가 작은 관측끼리 묶는다(object-centric).
- **삼각측량**: 2대 이상에서 보이면 마스크 대표점을 왜곡 보정한 뒤 가중 DLT로 계산한다. 소품 높이(카탈로그)를 써서 테이블 평면에 놓는다. 작업 공간(1.35×0.85×0.65m) 밖의 결과는 버린다.
- **개수 전제** (`keep_counted`): 이전 버전(v1.1)에서는 실시간으로 컵이 3개가 되거나, 오리가 공중에 뜨거나, 코끼리가 1↔2개로 깜빡였다. 그래서 "같은 소품은 목록 수만큼만 있다"는 강한 전제를 둔다.
- **상태와 사라짐**:
  - 상태는 candidate → active(여러 대에서 봄) 또는 partial(1대, 마지막 위치 유지)로 바뀐다.
  - 사라짐은 시간 기준 하나다: 1.5초 동안 안 보이면 사라진다. 처리 속도와 상관없이 같은 시간에 사라진다.

## 4. 방향 (서 있는 방향 yaw)

- **방법**:
  1. 추적 물체마다 카메라별 마스크로 사진을 잘라낸다.
  2. DINOv2-small 특징(고정)을 선형층에 넣어 24클래스 중 하나로 분류한다. 24클래스 = 자세 3 × 카메라 기준 방향 8칸.
  3. 카메라 방위각을 더해 테이블 기준 yaw로 바꾼다.
  4. 카메라 간 원형 평균을 내고, 추적 ID별로 누적한다(최근 비중 0.3). 0.5초마다 갱신한다.
- **표시 규칙**:
  - 서 있을 때만 트윈을 돌린다. 누움·뒤집힘은 `pose`로 기록만 한다.
  - 대칭(종이컵·머그)은 방향을 표시하지 않는다.
  - 에이전트가 돌린 물체는 에이전트 회전이 우선한다.
- **학습**: `tools/train_orientation.py --specs <specs.json> --raw <raw 폴더>` (합성 렌더의 정답 윤곽으로 자른 사진 사용).
- **비교 후보** (합성 40장면):

  | 후보 | 자세 정확도 | yaw 오차 | 시간 |
  |---|---|---|---|
  | 분류기 (채택) | 0.89 | 11° | 56ms |
  | 실루엣 맞추기 | 0.59 | 8° | 1.9초 |
  | VLM 판단 | 0.39 | — | 5.6초 |

  VLM 판단은 Meshy 모델의 "앞" 방향이 정해져 있지 않아 yaw로 바꿀 기준이 없었다.

## 5. 카메라 보정

- **내부 보정**: 같은 모델 카메라라 공용값(`calibration/creative_4cam`)을 재사용한다.
- **외부 보정**: A4 ChArUco 보드를 쓴다(5×3칸, 칸 56mm, 마커 42mm, DICT_4X4_50). 카메라마다 40장을 찍어 테이블 좌표에서 위치·자세를 구한다.
  - 예전 7×5칸·40mm 보드는 1.1m 거리에서 마커 한 칸이 2~3px라 1대만 읽었다.
- **테이블 찾기**: 4대 화면의 흰 상판을 보드 평면에 역투영하고, 크기를 아는 사각형(1.25×0.75m)을 맞춘다. 그래서 보드를 정중앙에 둘 필요가 없다.
- **2026-10-25 결과**: 재투영 0.10~0.14px, 40장 사이 위치 흔들림 1.3~9.3mm다. 이 숫자는 보정 품질이며 물체 위치 정확도가 아니다.
- **도구** (`VENUE_DATE=YYYYMMDD`): `print_charuco_board.py` → `venue_camera_preview.py` → `capture_venue_board.py` → `solve_board_extrinsics.py` → `overlay_venue_extrinsics.py`

## 6. 트윈과 장소 프리셋

- Isaac 확장(`gist.tabletop_twin`)이 UDP 메시지를 받아 소품 USD(`assets/props/<id>/model_placed.usda`)를 배치하고 yaw를 적용한다. 이름표는 부스 목록의 한국어 이름이다.
- 장소 프리셋(`config/venues.json`)은 배경 감싸기 USD, 테이블 위치(offset), 회전(yaw), 첫 시점을 묶은 것이다.
  - 기본 프리셋은 동신여고 AI 창의교실의 대회 사진 액자 벽 앞이다.
  - 실제 부스 장소는 따로 있다. 배경은 "학생 부스"를 보여 주는 무대 연출이다.
- 측정값: Isaac 앱 준비 20~25초, 프로세스 메모리 약 10GB.

## 7. 에이전트와 물리 실험

- 로컬 Gemma 4가 계획을 세우고, 코드(하네스)가 검증하고, PhysX가 계산한다. 자세한 내용은 `smartphone_mocap/EXPERIMENT_HARNESS.md`와 `PHYSICS_EXPERIMENTS.md`에 있다.
- **물리 경로로 가는 조건**: 요청에 "시뮬" 또는 "물리"라는 말이 있어야 한다. 예: "오리를 기린 쪽으로 0.1N의 힘으로 1초 동안 미는 시뮬레이션 세팅해줘".
- **실험 순서**:
  1. 모델은 `plan_experiment(changes)`로 바뀌는 것만 적는다.
  2. 코드가 단위·방향·대상을 검증한다. 힘(N)에 지속 시간이 없으면 실험을 만들지 않고 되묻는다.
  3. `physics_preview`로 실험을 일시정지 상태로 준비한다.
  4. `control_physics(play)`로 실행한다.
  5. `get_experiment_result`로 실제 PhysX 접촉 쌍을 확인한다.
- **단위는 섞지 않는다**: 힘 N, 충격량 N·s, 속도 m/s.

## 8. 평가 결과

**합성 120장면 비교** (`docs/eval/20261025_recognition_comparison.md`)

| 조합 | 정확히 1개 + 맞는 이름 | 시간 |
|---|---|---|
| A1 (채택) | 0.919 | 72ms |
| A1 + VLM | 0.915 | — |
| 검출기 + DINOv2 대조 | 0.596 | — |
| 검출기 + VLM | 0.749 | 3초 |
| COCO로 찾기 | 0.11 | — |

- 학습하지 않은 물건은 3~4%만 찾는다.
- 소품 검출기는 같은 방식의 렌더로 학습했으므로 이 평가에서 유리하다.

**실험실 실사** (2026-10-25~26)

| 시험 | 결과 |
|---|---|
| 배치 관찰 6분 | 8개 중 7개가 각각 1개씩, 카메라 3~4대에서 보임. 누운 기린·공룡은 "누움"으로 판정 |
| 움직임 시험 9분 | 1775프레임(약 3.2Hz). 같은 소품 2개는 3프레임. 들고 옮기면 0.3~1초 단위로 위치가 뛰며 따라감 |
| 자동 점검 (정지 30초 재인식) | 7종 모두 정확히 1개 99%, 위치 흔들림 0.1~1.0cm |

- 스테인리스 머그는 4대 모두에 보여도 검출되지 않는다.
- **측정하지 않은 것**: 위치 오차(cm), yaw 정확도, 치운 뒤 실제 사라지는 시간.

## 9. 한계

- 학습한 10종만 인식한다. 새 소품은 재학습해야 한다.
- 머그는 검출되지 않는다. 실물 사진으로 만든 모델로 재학습할 후보다.
- 처리가 약 3Hz라 빠른 움직임에서 위치가 튀고, 잠깐 2개로 보일 수 있다.
- 누움·뒤집힘은 표시하지 않는다.
- 물리 충돌 형상은 상자·원통·구 근사이고, 물성은 추정값이다. 결과는 정성적 설명용이다.
- GPU 하나를 Isaac 렌더링, 인식, 언어 모델이 함께 쓴다. 동시에 무거운 작업을 하면 화면이 느려지거나 Isaac이 멈출 수 있다.
