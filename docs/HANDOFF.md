# 인수인계 (사람용)

2026-10-26 기준. 이 데모를 처음 맡는 사람이 **켜고, 확인하고, 고장 났을 때 대응**할 수 있게 쓴 문서다.
- 설치: [`INSTALL.md`](INSTALL.md)
- 따로 받는 파일 넣기: [`ASSETS.md`](ASSETS.md)
- 원리와 학습: [`METHODOLOGY.md`](METHODOLOGY.md)
- AI 에이전트용 요약: [`HANDOFF_AI.md`](HANDOFF_AI.md)

## 1. 이 데모가 하는 일

1. 테이블(1.8×0.75m) 위 소품을 모서리 카메라 4대가 촬영한다.
2. 컴퓨터가 소품마다 이름, 3D 위치, 서 있는 방향을 계산한다.
3. Isaac Sim 화면(동신여고 AI 창의교실 배경)에 같은 소품의 3D 모델을 같은 자리에 띄운다.
4. 관람객은 한국어로 장면에 대해 묻거나 물리 실험을 시킬 수 있다.
   - 예: "오리를 기린 쪽으로 0.1N의 힘으로 1초 동안 미는 시뮬레이션 세팅해줘".
   - 물리 실험 요청에는 **"시뮬레이션" 또는 "물리"라는 말이 들어가야 한다.** "밀어 줘"만 말하면 일반 배치 요청으로 처리되어 "물리 기능이 없다"는 답이 나온다.

실제 부스는 동신여고 건물이 아닌 다른 장소다. 배경의 교실은 "학생 부스" 연출용이다.

## 2. 켜기

1. **조명을 켠다.** 어두우면 카메라 한 대만 테이블을 봐서 위치가 멈춘다.
2. **카메라 위치를 확인한다.** 카메라를 움직였으면 외부 보정을 다시 한다(6절).
3. **USB 대역폭을 적용한다.** 허브 하나를 두 대가 같이 쓰는 경우, 재부팅했거나 그 카메라를 다시 꽂았으면 다음을 실행한다.
   ```bash
   sudo bash tools/creative_uvc_bandwidth.sh apply
   ```
4. **소품 목록과 배경을 확인한다.**
   ```bash
   python3 tools/booth_check.py                       # 목록 점검
   python3 tools/booth_check.py --props               # 목록에 쓸 수 있는 소품(카탈로그)
   python3 tools/venue_args.py dongshin_ai_classroom   # 배경 파일 확인
   ```
5. **시작한다.**
   ```bash
   ./tools/booth_start.sh
   ```
   진행 단계는 세 가지다.
   - [1/3] 목록과 장소 점검
   - [2/3] 언어 모델 SGLang 준비. 처음에는 몇 분 걸린다.
   - [3/3] Isaac, 추적기, 에이전트 실행. Isaac 앱은 20~25초면 준비된다.
6. **정상 화면을 확인한다.**
   - 위쪽 뷰포트에 교실, 액자 벽, 테이블, 소품이 보인다.
   - 아래 패널에 카메라 4대의 검출 화면과 분할 화면이 각각 2×2로 나온다.
   - 오른쪽에 에이전트 대화창이 있다.
   - 화면이 비면 `Reset demo layout`을 누른다.
   - 뷰포트가 낮아 테이블이 화면 아래에서 잘리면, 시점 인자를 붙여 Isaac을 켠다(8절).

### 끄기

터미널에서 Ctrl+C를 누르거나 Isaac 창을 닫는다. 따로 끌 때는 `ps`로 PID를 확인하고 `kill -INT <PID>`로 끈다. 꺼지지 않으면 `kill -TERM <PID>`를 쓴다. `pkill -f`처럼 패턴으로 죽이지 않는다. 다른 셸까지 죽을 수 있다.

## 3. 처음 맡았을 때 하는 점검

점검은 두 단계다. 둘 다 "망가지지 않았는지"를 보는 것이고, 정확도 평가가 아니다.

**1) 자동 점검 (약 1분, 카메라 불필요)**

```bash
.venv-mocap/bin/python tools/booth_smoke_test.py
```

- **에이전트 점검**: 에이전트 서비스(7080)와 언어 모델이 켜져 있어야 한다. 실험실 장면 하나(`smartphone_mocap/data/smoke_scene.json`, 소품 7종)로 두 가지를 확인한다.
  - "테이블 위에 뭐가 있어?"에 소품을 모두 **한국어 이름으로** 답하는지
  - 시간이 빠진 힘 요청에 "몇 초 동안?"을 되묻는지
- **인식 점검**: 직접 찍은 녹화가 있을 때만 한다(4절에서 녹화를 만든다). 녹화 앞 30초를 다시 인식해 확인한다.
  ```bash
  .venv-mocap/bin/python tools/booth_smoke_test.py --session recordings/<녹화폴더> --expect giraffe zebra elephant
  ```
  - 녹화 앞 30초 동안 소품을 **가만히** 두어야 한다.
  - 통과 기준: 소품마다 "정확히 1개"가 90% 이상, 위치 흔들림 2cm 이하.
  - 놓았지만 아직 못 찾는 소품은 `--known-missing stainless_mug`처럼 알려 준다.
- **결과**: 실패가 하나라도 있으면 종료 코드 1이다. 결과는 `outputs/smoke_test/<시각>/report.json`에 남는다.
- **2026-10-26 실험실 결과**: 인식 7종 모두 통과(정확히 1개 99%, 흔들림 0.1~1.0cm). 에이전트 세 항목도 모두 통과했다.

**2) 현장 5분 점검 (카메라·Isaac 켠 상태)**

1. 소품 3개를 테이블에 올린다. 2초 안에 트윈에 맞는 이름표로 하나씩 나타나야 한다.
2. 하나를 손으로 20cm 옮긴다. 트윈이 따라와야 한다. 약 3Hz라 조금씩 끊기며 따라오는 것은 정상이다.
3. 하나를 테이블에서 치운다. 약 1.5초 뒤 트윈에서 사라져야 한다.
4. 에이전트 창에 "테이블 위에 뭐가 있어?"라고 묻는다. 올린 소품이 한국어 이름으로 모두 나와야 한다.
5. 물리 실험을 세 문장으로 이어서 보낸다.
   - "오리를 기린 쪽으로 0.1N의 힘으로 1초 동안 미는 시뮬레이션 세팅해줘": 일시정지 상태로 실험이 준비돼야 한다.
   - "실행해줘": 물리 재생이 시작돼야 한다.
   - "실험 결과에서 오리가 기린과 부딪혔는지 확인해줘": 실제 PhysX 접촉 기록으로 답해야 한다.
   - 실험 계획 생성과 실제 접촉은 Isaac이 있어야 하므로 이 단계에서만 확인된다.
6. 끝나면 대화창의 "새 대화"를 누르고, 실험을 초기화(reset)하거나 관측으로 돌아간다(return).

## 4. 녹화하고 다시 보기

녹화에는 카메라 4대 영상(AVI, 5fps)과 추적 기록이 함께 들어간다. 사람이 찍히므로 공개 저장소에 올리지 않는다(`recordings/`는 git 제외).

1. 녹화용 설정을 만든다. 부스 설정을 복사하고 `"recording": {"enabled": true, …}`로 바꾼다.
   ```bash
   cp smartphone_mocap/config/booth.json smartphone_mocap/config/booth_record.json
   ```
2. 녹화하면서 데모를 켠다.
   ```bash
   TABLETOP_CONFIG=smartphone_mocap/config/booth_record.json ./tools/run_tabletop_demo.sh
   ```
   녹화는 `recordings/<날짜_시각>_<라벨>/`에 생긴다. 끌 때는 추적기에 Ctrl+C를 보내야 녹화가 정상으로 마무리된다(`session.json`의 `completed: true`).
3. 카메라 없이 녹화로 트윈을 다시 본다.
   ```bash
   TABLETOP_CONFIG=smartphone_mocap/config/booth.json ./tools/run_isaac_tabletop_perception.sh   # Isaac만
   ./tools/run_tabletop_replay.sh recordings/<녹화폴더>                                          # 다른 터미널, 반복 재생
   ```
   재생은 녹화 영상을 지금 설정으로 **다시 인식**한다. 그래서 녹화할 때의 결과와 조금 다를 수 있다.

## 5. 소품 목록 바꾸기 (학생이 고치는 유일한 파일)

파일은 `config/booth_props.json`이다. 소품마다 아래 네 칸을 적는다.

- `prop_id`: 카탈로그(`assets/props/catalog.json`)에 있는 이름. 바꾸지 않는다.
- `name_ko`: 화면 이름표와 대화에 쓸 한국어 이름.
- `aliases_ko`: 같은 물건을 부르는 다른 말의 목록.
- `count`: 테이블에 올릴 개수(1 이상). 컵을 두 개 올리면 2로 한다.

고친 뒤에는 `python3 tools/booth_check.py`로 점검하고 데모를 다시 켠다. 오류 문구는 한국어로 나온다.

지금 목록(12종)은 두 가지 검출기가 나눠서 찾는다.

| 검출기 | 소품 | 특징 |
|---|---|---|
| 학습 소품 검출기(A1) | 기린, 얼룩말, 코끼리, 공룡, 오리, 로봇 | 우리 피규어로 학습. 서 있는 방향(yaw)도 표시 |
| 일반 물체 검출기(COCO 80종) | 컵, 휴대폰, 책, 화분, 페트병, 회색 마스코트(곰인형으로 찾음) | 학습 없이 씀. 관람객의 휴대폰·컵처럼 모양이 제각각이어도 찾음. 방향은 표시하지 않음 |

- 같은 물건을 두 검출기가 함께 잡으면 학습 소품 쪽만 남긴다.
- 새 물체는 카메라 2대 이상이 같이 잡거나, 1대만 잡아도 신뢰도 0.6 이상이면 테이블 면 위치로 생긴다(`object_detection.single_view`). 같은 종류 물체 15cm 안에는 새로 만들지 않는다.
- 학습 모델이 일반 물체도 알면 `checkpoint_classes`에 그 목록을 적는다. 거기 있는 소품은 COCO 검출기 대신 학습 모델이 찾는다(B안). 지금은 A안이다.
- 신뢰도 기준은 0.5다. 휴대폰만 놓치지 않도록 0.3으로 낮췄다(`booth.json`의 `object_detection.coco_class_confidence`). COCO 실사 검증에서 휴대폰 재현율이 0.53에서 0.75로 오르고 정밀도는 0.93에서 0.74로 내려간다.
- 카탈로그 항목의 `"detector": "coco"`와 `"coco_class"`(COCO 이름)가 일반 물체 검출기로 찾게 한다. COCO 80종에 있는 물건만 이렇게 추가할 수 있다(지갑·열쇠·안경은 없음).
- 그 밖의 새 물건은 재학습이 필요하다([`ASSETS.md`](ASSETS.md)의 "새 소품을 추가하려면").

## 6. 카메라 보정 (카메라를 옮겼거나 새 장소일 때)

저장소의 보정(`smartphone_mocap/calibration/venue_20261025`)은 2026-10-25 실험실 설치 기준이다. 다른 장소에서는 **반드시 새로 한다.**

카메라 배치는 다음과 같다(조작자 기준).

| 카메라 | 위치 | 형식(실험실) | 높이(실험실) |
|---|---|---|---|
| A | 뒤 왼쪽 | YUYV 10fps | 0.87m |
| B | 앞 왼쪽 | MJPG 15fps | 0.69m |
| C | 뒤 오른쪽 | YUYV 10fps | 0.64m |
| D | 앞 오른쪽 | MJPG 15fps | 0.88m |

조작자는 B와 D 사이의 긴 면 앞에 선다.

좌표계는 다음과 같다.
- 원점: B쪽 앞 왼쪽 모서리
- +X: B에서 D 방향
- +Y: 벽 쪽
- +Z: 위

보정 보드는 A4 한 장짜리 ChArUco다. 5×3칸, 칸 56mm, 마커 42mm, DICT_4X4_50이며 **100%로 인쇄한다.**

새 장소에서는 먼저 날짜 폴더를 만든다. 지난 설치 기준 `setup.json`(보드 규격·카메라 정보)을 복사해 새 장소에 맞게 고친다.

```bash
mkdir -p smartphone_mocap/calibration/venue_YYYYMMDD
cp smartphone_mocap/calibration/venue_20261025/setup.json smartphone_mocap/calibration/venue_YYYYMMDD/
VENUE_DATE=YYYYMMDD .venv-mocap/bin/python tools/print_charuco_board.py      # A4 PDF
VENUE_DATE=YYYYMMDD .venv-mocap/bin/python tools/venue_camera_preview.py     # 방향·코너 수 확인 (q로 닫기)
VENUE_DATE=YYYYMMDD .venv-mocap/bin/python tools/capture_venue_board.py      # 카메라마다 40장
VENUE_DATE=YYYYMMDD .venv-mocap/bin/python tools/solve_board_extrinsics.py   # cam_X.json + 보고서
VENUE_DATE=YYYYMMDD .venv-mocap/bin/python tools/overlay_venue_extrinsics.py # 확인 그림
```

- 보드를 테이블 정중앙에 둘 필요는 없다. 흰 상판에 크기를 아는 사각형을 맞춰 테이블을 찾기 때문이다.
- 10/25 결과: 재투영 오차 0.10~0.14px, 40장 사이 흔들림 1.3~9.3mm.
- 이 수치는 보정 품질이다. **물체 위치 정확도는 아니다.**
- 새 보정을 쓰려면 `smartphone_mocap/config/booth.json`의 `cameras[].calibration` 경로를 새 날짜 폴더로 바꾼다.

## 7. 학교에서 테이블로 (연출 버튼)

오른쪽 대화창 위의 **학교에서 테이블로** 버튼 하나로 한다.

- 동쪽 끝 가까이에서 잔디밭 너머로 건물 전체를 사선으로 보며 시작하고, 건물 정면을 따라 서쪽 교실 쪽으로 붙어 이동한다.
- 위에서 비스듬히 내려다보며 다가가는 동안 지붕·6층 → 5층 → 4층(홈베이스 포함) → 3층(교무실 포함) 순으로 위층이 사라진다. 한 층이 사라질 때 그 아래층의 천장·조명·보도 함께 사라져 각 층의 방 배치가 보인다.
- 마지막에 2층 천장·조명·에어컨까지 숨긴 "인형의 집" 모습으로 AI 창의교실에 내려와 테이블 앞에서 멈춘다(약 15초).
- 다시 누르면 숨긴 것을 모두 되살리고 학교 전경부터 다시 재생한다.

- 건물 전체를 그리는 동안은 화면이 느려질 수 있다. 손님이 없을 때 쓰는 것이 좋다. 인식·추적은 그동안에도 계속 돈다.
- 연출 계획은 `usd/venues/dongshin_ai_classroom_cutaway.json`이다. 카메라 경로나 숨길 층을 바꾸려면 `tools/make_cutaway.py`의 값을 고치고 Isaac python으로 다시 만든다.
- 손님이 없을 때 자동으로 한 번 재생하려면 Isaac 시작 인자에 `--/exts/gist.tabletop_twin/cutaway_autoplay_s=80`(시작 후 80초 뒤)을 붙인다.

## 8. Isaac 시점 조정

카메라 패널 때문에 뷰포트가 낮아지면, 장소 프리셋의 시점에서는 테이블이 아래로 잘릴 수 있다. 실행할 때 시점을 덮어쓴다(월드 좌표). 아래는 동신여고 교실 기준으로 2026-10-26에 쓴 값이다.

```bash
TABLETOP_CONFIG=smartphone_mocap/config/booth.json ./tools/run_isaac_tabletop_perception.sh \
  --/exts/gist.tabletop_twin/camera_eye_x=46.6 --/exts/gist.tabletop_twin/camera_eye_y=3.82 --/exts/gist.tabletop_twin/camera_eye_z=5.75 \
  --/exts/gist.tabletop_twin/camera_target_x=44.636 --/exts/gist.tabletop_twin/camera_target_y=3.82 --/exts/gist.tabletop_twin/camera_target_z=4.60
```

테이블만 가까이 보려면 eye를 `45.75, 3.82, 5.25`, target을 `44.636, 3.82, 4.55`로 준다. 설정 파일에 고정하려면 `config/venues.json`의 `camera`(테이블 좌표)를 바꾼다.

## 9. 고장 대응

| 증상 | 할 일 |
|---|---|
| 카메라 한 대가 안 나옴(0fps) | `sudo bash tools/creative_uvc_bandwidth.sh apply` |
| 물체가 멈춰 있거나 한 카메라에서만 잡힘 | 조명 확인. 카메라가 가려졌는지 확인 |
| Isaac이 켜지자마자 꺼짐("TSC ran backwards") | 일시적 오류다. 다시 켠다 |
| 운영 중 Isaac만 꺼짐(GPU 오류) | Isaac만 다시 켠다. 추적기 재시작은 가능하면 피한다 |
| 화면 일부가 비거나 레이아웃이 깨짐 | `Reset demo layout` |
| 트윈에 소품이 안 보이는데 이름표는 있음 | `assets/props/<id>/model.usdz`가 없음([`ASSETS.md`](ASSETS.md)) |
| `[문제] … 원본 USD가 없습니다` | 배경 파일을 넣지 않음([`ASSETS.md`](ASSETS.md)) |
| "학습한 소품이 아닙니다" 오류 | booth_props.json의 prop_id 오타. `booth_check.py --props`로 확인 |
| 에이전트가 "물리 기능이 없다"고 답함 | 요청에 "시뮬레이션" 또는 "물리"를 넣는다 |
| 언어 모델이 10분 안에 준비되지 않음 | `./tools/run_sglang_tabletop.sh logs` |
| 에이전트 코드를 고쳤는데 그대로임 | 에이전트 서비스를 다시 켠다. SIGINT로 안 꺼지면 SIGTERM을 보낸다 |

## 10. 꼭 지킬 것

- 실행 중에는 Isaac 확장 소스(`exts/`)를 고치지 않는다. 자동 다시 불러오기(hot reload)로 장면과 실험이 사라진다.
- 녹화, 실행 DB, 모델 가중치, Meshy 원본, 배경 USD, 인증정보를 git에 넣지 않는다.
- 결과를 말할 때 다음을 섞지 않는다.
  - 합성 수치와 현장 정확도
  - "개수·이름이 맞다"와 "위치가 맞다"
  - 모델이 낸 호출, 하네스 통과, Isaac 적용, 실제 접촉
- 물리 단위를 섞지 않는다. 힘은 N, 충격량은 N·s, 속도는 m/s다. 힘에 지속 시간이 없으면 에이전트가 되묻는 것이 정상이다.
