# SceneBridge — 현실에서 가상까지, 트윈의 과정을 체험

테이블 위 소품을 USB 카메라 4대로 보고, 각 소품이 **무엇인지, 어디에 있는지, 어느 쪽을 보는지** 계산한다. 그 결과를 Isaac Sim 가상 장면(디지털 트윈)에 실시간으로 옮기고, 트윈 안에서 한국어로 묻고 물리 실험까지 해 보는 데모다. 동신여고 학생 운영 부스용으로 만들었다(버전 v1.2).

```
카메라 4대 → 소품 인식(RF-DETR-Seg, 합성 데이터 학습) → 시점 연결·삼각측량 → 추적·방향
          → Isaac Sim 트윈 → 한국어 에이전트(로컬 Gemma 4) → PhysX 물리 실험
```

## 무엇이 들어 있나

| 경로 | 내용 |
|---|---|
| `smartphone_mocap/` | 추적기(인식·3D·추적·방향), 에이전트·물리 실험, 녹화·재생, 합성 데이터 도구 |
| `exts/gist.tabletop_twin/` | Isaac Sim 확장(트윈 화면, 카메라 패널, 에이전트 대화창) |
| `config/` | 부스 소품 목록(`booth_props.json`), 장소 프리셋(`venues.json`) |
| `assets/props/` | 소품 카탈로그와 배치용 USD(`model_placed.usda`). Meshy 원본은 따로 받음 |
| `usd/venues/` | 배경 감싸기 USD. 동신여고 원본 배경은 따로 받음 |
| `smartphone_mocap/calibration/` | 카메라 내부 보정, 2026-10-25 실험실 외부 보정 |
| `tools/` | 시작·점검·보정·학습 명령 |
| `tests/` | 회귀 테스트 |
| `docs/` | 설치, 자산 배치, 방법론, 인수인계, 평가 |

녹화 영상, 모델 가중치, Meshy 원본, 배경 USD는 저장소에 없다. 모델 가중치·Meshy 원본·배경 USD는 따로 전달받아 정해진 위치에 넣는다([`docs/ASSETS.md`](docs/ASSETS.md)).

## 빠른 시작

1. 설치: [`docs/INSTALL.md`](docs/INSTALL.md). Isaac Sim 6.0.1, Python venv, SGLang + Gemma 4를 설치한다.
2. 따로 받은 파일 넣기: [`docs/ASSETS.md`](docs/ASSETS.md)
   - 배경: `usd/venues/source/Dongshin_Final/`
   - 소품 모델: `assets/props/<prop_id>/model.usdz`
   - 모델 가중치: `outputs/models/`
3. 카메라 연결과 보정: [`docs/HANDOFF.md`](docs/HANDOFF.md) 6절
4. 확인과 시작:
   ```bash
   python3 tools/booth_check.py
   .venv-mocap/bin/python tools/booth_smoke_test.py
   ./tools/booth_start.sh
   ```

## 문서

- [`docs/HANDOFF.md`](docs/HANDOFF.md): 운영(켜기, 점검, 녹화, 보정, 고장 대응)
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): 방법론(합성 데이터 학습, 3D·추적·방향, 보정, 평가, 한계)
- [`docs/HANDOFF_AI.md`](docs/HANDOFF_AI.md), [`AGENTS.md`](AGENTS.md): AI 에이전트용 요약과 규칙
- [`docs/eval/20261025_recognition_comparison.md`](docs/eval/20261025_recognition_comparison.md): 인식 조합 비교(합성 120장면)
- [`smartphone_mocap/EXPERIMENT_HARNESS.md`](smartphone_mocap/EXPERIMENT_HARNESS.md), [`PHYSICS_EXPERIMENTS.md`](smartphone_mocap/PHYSICS_EXPERIMENTS.md): 물리 실험 계약

## 현재 상태와 한계 (2026-10-26)

- 실험실에서 학습 소품 7종이 각각 1개씩 트윈에 표시되고, 들고 옮기면 따라온다(약 3Hz).
- 스테인리스 머그는 검출되지 않는다.
- 위치 정확도(cm)는 아직 측정하지 않았다.
- 학습한 10종만 인식한다. 새 소품은 재학습이 필요하다.
- 합성 데이터 평가 수치는 현장 정확도가 아니다.
- NVIDIA GB10(aarch64)에서만 검증했다.

## 이용

라이선스를 따로 지정하지 않았다. 공개해 둔 것은 참고와 인수인계를 위해서다. 코드를 다시 쓰거나 고쳐서 배포하려면 저장소 소유자에게 문의한다.
