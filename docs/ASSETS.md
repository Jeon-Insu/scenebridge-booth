# 따로 전달받는 파일 넣기 (배경 USD, Meshy 소품 모델)

저장소에는 **코드, 설정, 배치용 USD**만 들어 있다. 아래 세 가지(배경, 소품 모델, 모델 가중치)는 용량, 초상권, 재배포 조건 때문에 따로 전달받아 정해진 위치에 넣는다. 넣는 위치는 `.gitignore`에 등록되어 있어서 실수로 커밋되지 않는다.

## 1. 동신여고 배경 (Isaac 트윈의 무대)

전달받는 것: `Dongshin_Final` 폴더 전체(2026-10-27 최종 배포본, 약 658MB. 교무실·AI 교실·4층 홈베이스·복도 포함). 폴더 안의 `Dongshin_Final.usda`와 그 옆 텍스처·하위 USD를 모두 받아야 한다.

넣는 위치(폴더 이름 그대로):

```
usd/venues/source/Dongshin_Final/Dongshin_Final.usda
usd/venues/source/Dongshin_Final/...        ← 함께 온 파일 전부
```

- `usd/venues/dongshin_ai_classroom.usda`(저장소에 있음)가 이 파일을 `./source/Dongshin_Final/Dongshin_Final.usda`로 참조한다.
  - 참조만 하고 원본은 고치지 않는다.
  - 테이블 주변 책상·의자 28개를 화면에서 끈다. 3층 교무실은 보이게 두고, 연출 버튼이 3층과 함께 숨긴다.
- 확인 방법:
  ```bash
  python3 tools/venue_args.py dongshin_ai_classroom
  ```
  - 인자 4줄 이상이 나오면 정상이다.
  - 파일이 없으면 `[문제] … 원본 USD가 없습니다: <경로>`가 나온다.
- 배경이 없어도 **추적기·에이전트·점검 도구는 동작한다.** Isaac 트윈만 켜지지 않는다.
- 다른 공간을 배경으로 쓰려면 다음 순서로 한다.
  1. `config/venues.json`에 프리셋을 추가한다. 형식은 기존 항목과 같다: `background`, `offset_m`, `yaw_deg`, `camera`.
  2. 부스 설정 `smartphone_mocap/config/booth.json`의 `"venue"`를 그 이름으로 바꾼다.
- 테이블 자리를 다시 잡아 배경 감싸기 파일을 다시 만들 때(Isaac python):
  ```bash
  $ISAAC_SIM_DIR/python.sh tools/make_venue.py
  ```
  출력된 `offset_m`, `yaw_deg`를 `config/venues.json`에 넣는다.

## 2. 소품 3D 모델 (Meshy에서 만든 USDZ)

전달받는 것: Meshy로 만든 소품 모델 `model.usdz` (학습 소품 10종 + 화분 + 회색 마스코트). Meshy로 만들고 Meshy 형식(Y-up, 단위 0.3048m) 그대로 내보낸 파일이다.

넣는 위치는 **`assets/props/<prop_id>/model.usdz`**이다. 파일 이름은 반드시 `model.usdz`다.

| prop_id (폴더 이름) | 한국어 이름 | 넣을 파일 |
|---|---|---|
| `gist_mascot` | 마스코트 | `assets/props/gist_mascot/model.usdz` |
| `giraffe` | 기린 | `assets/props/giraffe/model.usdz` |
| `zebra` | 얼룩말 | `assets/props/zebra/model.usdz` |
| `elephant` | 코끼리 | `assets/props/elephant/model.usdz` |
| `dinosaur` | 공룡 | `assets/props/dinosaur/model.usdz` |
| `devil_duck` | 오리 | `assets/props/devil_duck/model.usdz` |
| `green_robot` | 로봇 | `assets/props/green_robot/model.usdz` |
| `desk_clock` | 시계 | `assets/props/desk_clock/model.usdz` |
| `paper_cup` | 종이컵 | `assets/props/paper_cup/model.usdz` |
| `stainless_mug` | 머그 | `assets/props/stainless_mug/model.usdz` |
| `potted_plant` | 화분 | `assets/props/potted_plant/model.usdz` |
| `gist_mascot_grey` | 회색 마스코트 | `assets/props/gist_mascot_grey/model.usdz` |

휴대폰·책·페트병은 간단한 형상 USD가 저장소에 들어 있어 따로 받을 것이 없다. 컵은 종이컵 모델을 함께 쓴다.

- **바로 옆 `model_placed.usda`(저장소에 있음)가 `./model.usdz`를 참조한다.**
  - 이 파일이 Meshy 좌표를 테이블 좌표(Z-up, 미터, 바닥 중심 원점)로 바꾸고 크기를 맞춘다.
  - 그래서 **전달받은 Meshy 파일을 바꾸지 않고 그대로** 넣어야 한다. 다른 Meshy 모델로 바꾸면 크기와 바닥 위치가 어긋난다.
- 모델이 없으면 트윈에서 그 소품이 보이지 않는다(참조 실패). 인식과 추적에는 영향이 없다. 인식은 학습된 검출기가 하기 때문이다.
- 확인 방법: `ls assets/props/*/model.usdz`로 12개가 있는지 본다.

### 새 소품을 추가하려면

새 소품은 인식 모델이 학습한 적이 없어서 **재학습이 필요하다.** 학습하지 않은 물건은 약 3~4%만 찾는다. 순서는 다음과 같다.

1. 이름을 정한다. `prop_id`는 영문 소문자와 밑줄을 쓴다(예: `red_car`).
2. Meshy에서 만든 모델을 `assets/props/<prop_id>/model.usdz`에 넣는다.
3. `assets/props/catalog.json`에 항목을 추가한다. `asset_id`, `display_name_ko`, `usd_uri`=`<prop_id>/model_placed.usda`, `height_m`을 적는다.
4. `model_placed.usda`를 만든다. `bash tools/normalize_prop_assets.sh`를 실행하면 된다. Isaac Sim의 USD 라이브러리를 쓰고, 이미 만든 소품은 건너뛴다. 그다음 실물 높이를 `python3 tools/set_prop_height.py <prop_id> <높이m>`로 맞춘다.
5. 합성 데이터를 만들고 검출기를 재학습한다. 방법은 `docs/METHODOLOGY.md`의 2절에 있다.
6. 부스 목록 `config/booth_props.json`에 넣고 `python3 tools/booth_check.py`로 확인한다.

## 3. 모델 가중치

이 둘도 따로 전달받는다. 이름을 바꾸지 말고 아래 위치에 넣는다. 없으면 추적기가 시작하지 않는다.

- `outputs/models/prop_detector_a1.pth`: 소품 검출기, 128MB
- `outputs/models/orientation_head.pt`: 방향 분류기

## 넣지 않는 것

- 녹화 영상: 사람이 찍혀 있어 공개하지 않는다. 직접 녹화하는 방법은 `docs/HANDOFF.md`에 있다.
- 실행 DB, 로그, `outputs/` 아래 결과.
- 인증정보: Meshy API 키는 `.secrets/meshy_api_key` 또는 환경변수 `MESHY_API_KEY`에만 둔다.
