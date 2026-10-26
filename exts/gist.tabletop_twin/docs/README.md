# GIST Tabletop Twin

2026-09-14 추가: [미등록 객체 · Meshy 3D 생성 · 자산 적용](../../../smartphone_mocap/ASSET_STUDIO.md). 아래의 기존 `approved_waiting_provider` 설명은 이전 흐름이며, 현재 입력 선택·생성 화면은 새 문서를 기준으로 사용합니다.

Demo 2용 사용자 인터페이스 Extension이다. 기존 외부 카메라 프로세스가 보내는
`gist.smartphone_mocap.v1` UDP 메시지에서 객체만 받아 테이블 좌표계에 표시한다.

주요 기능:

- 왼쪽 상단의 큰 Isaac Viewport 유지
- 오른쪽 전체 높이를 사용하는 `Tabletop Agent | Stage` 탭 (Agent 기본 선택)
- Viewport 아래 `Tabletop Cameras` **한 탭** 안에 왼쪽 검출 2×2 + 오른쪽 분할 2×2 동시 표시
- 같은 하단 영역의 `Tabletop Settings`, `Content`, `Console` 탭; Property는 기본 숨김
- Layer·Render Settings·Robot Inspector도 기본 숨김. 초기 레이아웃 복원이 끝날 때까지
  잠시 재등장을 억제해 Viewport가 바로 보이도록 한다. 필요하면 시작 완료 후 Window 메뉴로 다시 연다.
- 대화 기록은 남는 높이를 사용하며 최대 200개 메시지를 줄바꿈·스크롤로 표시
- 1.25 × 0.75 × 0.75 m 기준 테이블 자동 로드
- 추천 테이블 소품 6종 카탈로그 표시
- 광주 옛전남도청 OSM 100 m 배경 장면 합성 및 시작 시 Viewport 자동 맞춤
- 감지 객체별 USD Prim 생성·선택 및 30초 후 stale Prim 삭제
- 모델이 없는 객체의 3D 재구성 요청과 사용자 승인 대기열
- `LIVE → SNAPSHOT → AGENT → RESUME` 로컬 장면 Agent와 상태 출력

### 객체 선택과 카메라 강조 (0.6.0)

Stage/Viewport에서 등록된 객체(Asset 내부 mesh/손잡이 등 하위 Prim 포함)를 선택하거나
Agent의 `select_object`를 실행하면, 동일한 tracker ID로 연결된 실물 관측을 함께 강조한다.
카메라 패널 위의 `Cup 01`, `Cup 02`, `Plant 01` 등 짧은 이름 버튼으로도 선택할 수 있다.
대표 썸네일은 현재 프리뷰에서 선택 객체가 가장 크게 나온 연결 뷰를 잘라 표시한 것으로,
장기 외형 갤러리나 새로운 Re-ID 판단은 아니다. 짧은 이름은 실행 세션 동안 유지하고
삭제된 다른 ID에 재할당하지 않는다. 앱/Stage 재시작 시 새로 부여한다.

- 왼쪽 RAW+검출 2×2: 선택한 객체에 황금색 box, 나머지 영역은 약하게 어둡게 표시.
- 오른쪽 segmentation 2×2: 선택 instance의 mask를 강조하고 황금색 윤곽 표시. mask가 없으면 box 사용.
- 4대 모두 **상태는 표시**하지만, 정확히 한 검출과 연결된 뷰에만 실제 강조를 한다.
  `NOT OBSERVED`는 해당 프레임에서 연결된 검출 없음(가림/미검출/화면 밖 등을 구분하지 않음),
  `AMBIGUOUS LINK`는 같은 ID에 여러 검출이 연결됨, `NO FRAME`은 영상 없음이다.
  `LINK UNAVAILABLE`은 프리뷰와 연결 메타데이터의 일치를 검증하지 못한 상태다.
- 클래스가 같다는 이유로 다른 컵을 강조하거나, 가상의 이동된 좌표를 실물 영상에 투영하지 않는다.
  Ghost preview를 선택하면 원래 객체의 실물 관측을 강조한다.
- 선택 해제/다른 객체 선택은 녹화가 고정된 상태에서도 즉시 UI에 반영된다.
- 선택된 ID와 짧은 이름은 Agent snapshot에도 포함되어 `선택한 객체`/`Cup 02`를 지정할 수 있다.
  대상이 여러 개면 기존 유일성 검증에 따라 선택/수정을 거절하거나 확인을 요청한다.
- 이번 연결은 Stage/Viewport/Agent/이름 버튼 → 카메라 강조 방향이다.
  카메라 영상의 픽셀을 클릭해서 객체를 선택하는 역방향 기능은 아직 추가하지 않았다.

추론 프로세스가 `tabletop_camera_mosaic.links.json`에 카메라 순서, 같은 프레임의
tracker ID별 box/프리뷰 해상도 이진 mask, JPEG SHA-256을 함께 발행한다. UI는 JPEG와
메타데이터 해시가 일치할 때만 연결을 사용한다. 실제 라이브 입력과 녹화 replay 모두 지원한다.
선택 효과는 UI 전용 복사본에만 그린다. 원본/프리뷰 JPEG, Agent·재구성 입력은 수정하지 않는다.
이는 **기존 ID 연결 결과의 시각화**이며 카메라 간 객체 매칭 정확도를 개선하는 기능은 아니다.

실제 앱 검증:
`TABLETOP_UI_CHECK=1 TABLETOP_EVIDENCE_CHECK=1 TABLETOP_FOCUS_CHECK=1 TABLETOP_FOCUS_AGENT_CHECK=1 TABLETOP_REPLAY_START=1500 ./tools/run_tabletop_recorded_demo.sh`
컵 두 개의 선택 전환, 하위 mesh 선택, 양쪽 4뷰 ID 연결, 고정 화면에서 선택 해제,
모델 입력 JPEG 무변경, 로컬 Agent의 짧은 이름 선택을 확인한다.
2026-09-08 검증: Python 테스트 124개 및 headless USD 회귀 통과. Full App에서 위 선택
시나리오도 통과했으며 `outputs/validation/tabletop_ui_report.json`의 `camera_focus`에
카메라별 연결 결과를 기록했다. 당시 Cup 01/02 모두 3개 뷰에 ID가 연결됐고 C920은
연결되지 않아 `NOT OBSERVED`로 표시했다. 이는 C920 영상에 컵이 없다는 뜻이 아니다.

`Settings`에는 연결 상태, detector/mask 방식, 테이블 선택, 객체 초기화 및 객체별
XYZ·Z 계산 모드·사용 카메라 수·USD 선택·3D 생성 요청을 표시한다. `Agent`에서 처음
질문을 보내면 마지막으로 완전히 처리된 네 카메라 묶음과 bbox, instance mask, 객체 ID와
XYZ를 하나의 snapshot으로 고정한다. live 입력의 USB capture thread는 계속 실행하지만
RF-DETR/segmentation과 USD 갱신은 멈추므로 LLM에 GPU 자원을 돌릴 수 있다. 녹화 replay는
재생 시각 자체가 정지한다.

후속 질문은 같은 snapshot을 사용한다. `Refresh Scene`은 새 프레임을 한 번 처리해
snapshot을 교체하고, `Resume LIVE`는 실시간 추론을 재개한다. 일반 이동 제안은
`/World/AgentPreview` 아래의 cyan ghost로 표시한다. `적용해줘` 또는 `미리보기 적용` 버튼은
그 미리보기를 실제 USD 객체에 반영한다. `cup/.../instance_000을 동쪽으로 10cm 바로 이동해줘`
처럼 명시적으로 바로 이동을 요청하면 미리보기 없이 적용한다. 실제 물체나 카메라 영상은
변하지 않는다. 가상 객체는 추적 좌표 + 객체별 이동량으로 표시되므로 인식 재개 후에도
이동량을 유지한다. `추적 위치로 복귀해줘` / `추적 위치 복귀` 버튼으로 초기화한다.
이동량은 세션 내 해당 tracker ID에만 적용하며 객체가 제거되거나 extension이 재시작되면
초기화된다. 후속 장면 질의는 가상 XYZ와 카메라 관측 XYZ를 구별한다.

Viewport 왼쪽 위 `TABLE COMPASS`는 테이블 기준 N(+Y), E(+X), S(-Y), W(-X)를 표시한다.
카메라를 돌리면 표시 방향도 함께 바뀐다. 이 방향은 데모 TableFrame의 약속이며 실제
지리적 북쪽을 새로 측정하는 기능은 아니다. 화면 좌/우 이동은 질문 전송 순간의 카메라
기준이며, 동/서/남/북 이동은 고정 TableFrame 기준이다.

높이 명령은 `change_height`를 사용한다. `위로 30cm 올려줘` / `Z축으로 30cm 들어 올려줘`는
현재 가상 높이에 0.30m를 더하고, `10cm 내려줘`는 0.10m를 뺀다. `높이를 30cm로 맞춰줘`는
절대 원점 높이 Z=0.30m다. `테이블 위에 내려놔`는 별도 `place_on_table`로 형상의 하단을
Z=0에 둔다. X/Y는 유지한다. `change_height`가 설정하는 것은 테이블 기준 애셋 원점 Z이며,
기울어진 형상의 실제 하단 간격과 다르다. 요청한 원점 Z는 최소 0으로 제한하지만
이것만으로 형상 관통이 방지되지는 않는다. `화면 위쪽으로 이동`이라고 명시할 때만 XY 화면 방향으로
해석한다. 높이 명령은 기본적으로 즉시 적용하고 `미리보기`를 요청하면 ghost만 만든다.
물체 간 충돌·다른 물체 위에 쌓는 배치는 아직 계산하지 않는다.

회전과 크기 변경도 function call로 처리한다.

- `rotate_object(query, angle_deg, axis="z", mode="relative", preview_only=false)`:
  테이블의 고정 X/Y/Z축을 기준으로 quaternion을 합성한다. `Z축으로 90도 회전해줘`는
  현재 자세에 회전을 더하고, `Z축 회전을 90도로 맞춰줘`는 해당 축·각도로 전체 자세를
  재설정한다. Z축 시계 방향은 위에서 내려다볼 때 음수, 반시계는 양수다.
- `scale_object(query, factor, axis="uniform", mode="relative", preview_only=false)`:
  전체 또는 객체의 로컬 X/Y/Z 크기를 변경한다. `현재보다 20% 크게`는 현재 크기에 1.2배,
  `크기를 1.5배로 맞춰줘`는 원본 USD 기준 1.5배다. `높이 방향으로만 2배 늘려줘`는
  객체의 로컬 Z 스케일만 변경하며 객체의 위치는 움직이지 않는다.
- `reset_transform(query, components=["translation", "rotation", "scale"])`:
  원하는 항목만 초기화한다. `회전 초기화`, `크기 초기화`와 `회전/크기 초기화` 버튼을 지원한다.

`미리보기`를 요청하면 변환된 ghost를 표시하고 `미리보기 적용` 버튼 또는 `적용해줘`로 확정한다.
일반 회전·크기 명령은 즉시 적용한다. 원점(pivot)은 애셋에 작성된 원점이며 추천 소품은
대체로 바닥 중심이다. 기본값은 위치 유지이며, LLM이 `placement` 옵션으로 접촉 유지
또는 테이블 배치를 선택했을 때만 USD 경계를 계산해 높이를 보정한다. 자세한 정책과
향후 물리 시뮬레이션 확장 계획은 [placement-design.md](placement-design.md)에 기록했다.
`inspect_table_contact`는 접촉/간격/관통 가능성을 조회하고, `place_on_table`은 현재
회전·크기를 반영해 내려놓는다. `transform_object`는 회전+전체 크기를 한 번에 처리한다.
공중 객체의 회전은 위치 유지가 기본이다. 관통 가능성은 경고만 하며 자동으로 붙이지 않는다.
현재 검사는 보수적인 형상 경계 기반이고, 물리적 안정성이나 객체 간 충돌을 계산하지 않는다.
변환은 객체 ID별로 보관해 새 추적 패킷에서도 동일한 값으로 적용한다. 이는 카메라가
물체의 회전이나 크기를 측정한 값이 아니라 가상 편집값이다. 객체 삭제/세션 재시작 시
초기화되며, `list_objects`와 `scene_state.json`에 현재 quaternion(w,x,y,z)과 scale을 포함한다.

Isaac Sim은 `127.0.0.1:7080`의 가벼운 Agent bridge에만 접속한다. bridge는
`127.0.0.1:30000/v1`의 SGLang OpenAI-compatible API를 우선 사용하고, SGLang이 꺼져도
동작 검증이 가능한 규칙 기반 fallback을 제공한다. `run_tabletop_demo.sh`와
`run_tabletop_recorded_demo.sh`가 bridge를 자동으로 함께 시작한다.

3D 재구성 승인 버튼은 현재 외부 유료 API를 호출하지 않는다. 승인 상태를
`approved_waiting_provider`로 기록하므로 Meshy 또는 VARCO 제공자를 나중에 별도
어댑터로 연결할 수 있다.

DGX Spark에서 로컬 SGLang 서버를 준비하고 관리하는 명령은 다음과 같다. 기본값은
Gemma 4 26B-A4B NVFP4이며, 첫 실행은 container와 약 19 GB model을 내려받으므로 시간이
걸린다.

```bash
./tools/run_sglang_tabletop.sh start
./tools/run_sglang_tabletop.sh switch
./tools/run_sglang_tabletop.sh status
./tools/run_sglang_tabletop.sh logs
./tools/run_sglang_tabletop.sh stop
```

기본 model은 `nvidia/Gemma-4-26B-A4B-NVFP4`, endpoint는 30000번 포트다. 이 모델은
26B 전체 중 약 4B parameter만 token마다 활성화되는 MoE라 31B dense보다 부스 Agent의
응답 지연과 자원 사용이 낮다. 기본 26B-A4B 프로필은 동시 요청 2개와 총 KV token 131,072개로 제한해 Isaac Sim 및
vision pipeline과 메모리를 나눠 쓴다. Agent bridge는 `/v1/models`에서 실제 model을 자동으로
찾으므로 profile을 바꿔도 별도 JSON 수정이 필요 없다.

```bash
# Recommended multimodal MoE
SGLANG_PROFILE=gemma4-26b-a4b ./tools/run_sglang_tabletop.sh switch

# Higher-cost dense comparison
SGLANG_PROFILE=gemma4-31b ./tools/run_sglang_tabletop.sh switch

# Lightweight text-only fallback
SGLANG_PROFILE=qwen3-8b ./tools/run_sglang_tabletop.sh switch
```

Gemma 4 profile은 처음에 구조화된 상태 요약과 정보원 목록만 받는다. Agent가 필요한
실물 카메라 또는 가상 Viewport 이미지를 `inspect_images`로 요청했을 때 base64 이미지와
이미지별 설명을 전달한다. 이미지가 없거나 text-only model이면 조회 불가를 명시하며
구조화된 정보만 사용할 수 있다. 공개 model 접근에 token이 필요하지 않으면 그대로 실행하고,
Hugging Face가 인증을 요구할 때만 실행 전에 `HF_TOKEN`을 환경변수로 제공한다. 인식
파이프라인과 bridge는 SGLang container의 수명과 분리되어 있다.

고정 입력은 `outputs/tabletop_agent/snapshots/<snapshot-id>/`에 저장된다. 여기에는
`snapshot.json`, 카메라별 `raw/detection/segmentation.jpg`, `camera_mosaic.jpg`가 있다.
현재 제어 상태는 `outputs/tabletop_agent/control.json`과 `runtime_state.json`에서 확인한다.

이 Extension과 `gist.smartphone_mocap`은 기본 UDP 포트 7001을 공유하므로 Demo 2
실행 시에는 둘을 동시에 활성화하지 않는다.

추천 소품 배치만 먼저 확인하려면
`usd/tabletop_demo/decor_preview_scene.usda`를 연다. 실제 라이브 장면은 중복을
피하기 위해 소품을 미리 배치하지 않고 감지된 객체만 생성한다.

녹화된 네 카메라 입력을 반복 재생하면서 옛전남도청 배경 위에서 확인하려면 프로젝트
루트에서 다음 명령을 실행한다.

```bash
./tools/run_tabletop_recorded_demo.sh
```

이 실행기는 `usd/tabletop_demo/gwangju_osm_100m_tabletop.usda`를 열고, 기존
`usd/osm_100m/gwangju_scene.usda`를 sublayer로 사용한다. 물리 테이블은 옛 도청 건물
북쪽의 열린 공간에 배치된다. `Tabletop Cameras` 한 탭의 왼쪽은 원본 위의
bbox·class·XYZ 4뷰, 오른쪽은 bbox 없이 instance별 mask 영역을 진한 색으로 채운
4뷰를 각각 2×2로 동시에 표시한다. 양쪽은 **동일한 완성 모자이크**를 재배치하므로
별도 추론이나 다른 시각의 프레임을 추가하지 않는다. 카메라 순서는 양쪽 모두 첫 두
카메라가 위, 나머지 두 카메라가 아래다. 카메라 영상과 트윈 XYZ는 기존 추론 완료
주기로 갱신된다. 시작 시 Viewport는
배경 전체가 아니라 테이블을 자동으로 확대하며, 구도가 바뀌면 Settings의
`Reset Table View` 버튼으로 다시 맞출 수 있다.

기본 비율은 오른쪽 Agent 약 30%, 왼쪽 하단 약 35%다. Agent에는 입력·전송,
실시간 재개·미리보기 적용만 남긴다. 장면 갱신, Ghost 취소, 변환 초기화,
예시 질문과 3D 복원 승인은 `Tabletop Settings`에 있다. 대화 기록은 일반 상태 갱신 시
입력 위젯을 다시 만들지 않으므로 입력 중 포커스를 유지한다. 아래를 보고 있을 때 새
메시지를 따라가고, 과거 기록을 읽고 있을 때는 스크롤 위치를 유지한다.
배치를 복구하려면 Settings의 `Reset Demo Layout`을 누른다.

GUI 회귀 검증: `TABLETOP_UI_CHECK=1 ./tools/run_tabletop_recorded_demo.sh`.
검증은 도킹·대화창 높이·입력 위젯 유지·긴 한글 줄바꿈/스크롤을 확인하며, 임시 테스트
메시지는 끝나면 제거한다. 결과와 화면은 `outputs/validation/tabletop_ui_*`에 기록한다.

### LLM 이미지 입력과 3D 생성 입력은 별개

2026-09-08: 자동 모자이크 첨부를 **Agent 선택형 정보 조회**로 교체했다.

- `inspect_scene(source="physical"|"twin", query=...)`: 실물 추적 목록과 실제 등록 USD 상태를 분리한다.
- `spatial_relations(source, reference_query)`: 유일한 기준 객체부터 다른 객체까지 XYZ 차이와
  거리(원점 간, 표면 간격 아님)를 계산한다. `옆`의 거리 임계값은 임의로 고정하지 않는다.
- `inspect_images(source="physical", camera_ids=[...], kind="raw")`: 요청한 원본/검출/분할
  시점을 첨부한다. ID 생략 시 최대 4장. 각 이미지 **바로 뒤에** ID·코너·위치·방향·시각·검출 목록을 붙인다.
- `inspect_images(source="twin")`: 질문에 대응하는 고정 Viewport 이미지와 USD ID별 화면 투영 원점을 제공한다.
  투영 원점은 가림 검사를 하지 않으며 화면 안에 있다고 실제 보이는 것은 아니다.

질문마다 조회 순서를 강제하지 않는다. 실물/트윈 출처 지정, 불확실성, 이미지 바이트 예산,
stable ID 확인 및 재구성 승인 조건은 코드로 제한한다. 최근 대화 맥락도 없으면 설명 질문은
실물 기준으로 답하도록 안내하며, 답변과 UI의 정보원 표시로 기준을 알린다. 기본 최대 6회 tool round
뒤에는 추가 도구 없이 답변하는 기회를 준다. 카메라 누락을 가상 좌표로 메우지 않는다.
검출 목록은 완전한 객체 목록이 아니다. 이미지에서만 보이는 물체는 **미확정 시각 후보**로
설명할 수 있으나 아직 후보 레지스트리/영역 선택 UI가 없으므로 그 물체의 이동·재구성은 불가능하다.

실물은 고정 snapshot을 재사용하고 트윈은 질문마다 새로 캡처한다. USD 상태·렌더는 서로 같은
트윈을 설명하며 실제 공간에 대한 독립적인 검증이 아니다. 조회 도구는 큐에 넣은 가상 편집의
적용 전 상태를 반환한다. 적용 확인 후 다음 질문에서 새 캡처로 결과를 확인한다.
`outputs/tabletop_agent/requests/<query-id>/evidence.json`과 `viewport.png`에 질문별 입력을 보존한다.
캡처 중 등록 객체 상태나 가상 카메라가 바뀌면 해당 이미지는 조회 불가로 처리한다.
현재는 기존 Viewport만 캡처하며 위에서 내려다보는 추가 가상 카메라 자동 생성은 하지 않는다.
모델에 이미지를 보내는 것은 선택형이지만, 일관된 질문 시점을 위해 Viewport 파일 캡처는 질문마다 한 번 수행한다.
실물 카메라를 새로 열거나 사용자 Viewport의 각도를 바꾸지 않는다.

Gemma 26B-A4B 기본 프로필의 문맥 한도는 65,536 tokens다. 2026-09-14에 16K에서 64K로 확대하고 전체 KV 풀도 128K로 늘렸다. 물리 계획에서는 성공적으로 읽은 이미지를 다음 도구 단계마다 반복 첨부하지 않는다. 모델 파일은 동일하다.

검증: `TABLETOP_UI_CHECK=1 TABLETOP_EVIDENCE_CHECK=1 TABLETOP_REPLAY_START=1500 ./tools/run_tabletop_recorded_demo.sh`
는 실물 snapshot 고정·4카메라 메타데이터·USD/Viewport 버전 일치까지 확인하고, 검증용
`outputs/validation/tabletop_evidence_snapshot.json`을 생성한다. 이 옵션은 replay를 고정 상태로 둔다.
`TABLETOP_REPLAY_START`는 검증 시 물건이 배치된 녹화 구간을 선택하는 옵션이며 기본값은 0이다.
`TABLETOP_AGENT_UI_CHECK=1`도 지정하면 UI에서 읽기 전용 비교 질문을 실제로 보내,
실물 이미지와 가상 이미지 조회·출처 표시·무편집 완료까지 검증한다.
` .venv-mocap/bin/python tools/validate_tabletop_agent_evidence.py`는 저장된 검증 snapshot으로
실물 개요/가상 화면/실물-트윈 거리 비교 3개 질문을 로컬 모델에 보낸다. 반환된 action은 Isaac에 전송하지 않는다.

검증 기록(2026-09-08): Python 테스트 110개 통과, headless USD 변환/접촉 회귀 통과,
Full App의 도킹·고정 이미지/좌표·대화 UI 종단 경로 통과. 로컬 모델이 선택한 조회도
3개 질문 모두 통과했다. 한 번의 측정은 실물 4장 개요 17.20초, 가상 화면 7.12초,
양쪽 좌표 비교 12.24초였으며 일반적인 지연/인식 정확도 보장은 아니다.
검증은 정보원 선택과 데이터 전달을 확인한다. 시각적 누락 판정·동일 객체 대응·캘리브레이션
정확도는 별도 평가 대상이다. 특히 '추가 물체를 발견하지 못함'은 실제 누락이 없다는 증명이 아니다.

3D 생성 API는 아직 연결하지 않았다. 현재는 `awaiting_approval` →
`approved_waiting_provider` 상태만 있다. 공급자·키·테스트 대상을 정한 뒤,
대상 객체의 원본 crop(박스·색상 mask 오버레이 제외)을 검토/승인하고 전송해야 한다.
생성 결과의 크기·축·원점·재질을 검증한 후 별도 asset으로 등록하고 기존 tracker ID와 연결한다.
실시간 루프에서 중복 생성하거나 매번 과금되지 않도록 작업 ID·실패·재시도 관리가 필요하다.
[Meshy 공식 다중 이미지 API](https://docs.meshy.ai/en/api/multi-image-to-3d)는 같은 객체의
서로 다른 각도 이미지 1~4장을 받는다. 실제 유료 생성 호출은 아직 수행하지 않았다.

[SGLang](https://docs.sglang.io/docs/basic_usage/openai_api_vision)은 지원 모델에서 한 요청에
여러 이미지와 설명을 교대로 넣을 수 있다. 2026-09-08 현재 Gemma 구성에 원본 4장과
camera ID/코너 설명을 한 요청으로 보내는 별도 읽기 전용 테스트가 성공했다. 한 번의
측정은 13.68초, 입력 1,345 tokens(그중 image 1,056), 출력 153 tokens였다. 이는
일반적인 성능 보장이 아니며 정확도 비교도 아니다. 현재는 위의 선택형 입력을 사용한다.

3D 입력 준비는 `동일 snapshot 묶음 → global object ID 매칭 → 원본 crop → 해당
instance의 이진 mask로 배경 제거 → 품질 검사 → 사용자 승인 → API` 순서로 확장한다.
현재 스냅샷은 원본·검출·색칠된 분할 JPEG와 bbox/mask 면적을 보관하지만, 객체별 이진
3D 생성용 원본 해상도의 이진 mask는 아직 저장하지 않는다(UI 강조용 프리뷰 해상도 mask는 별도).
카메라별 timestamp/보정 offset/프레임 번호/시점 위치는
현재 snapshot의 `camera_metadata`에 저장하고 `timing.camera_skew_ms`에 선택 프레임 시각 차이를 기록한다.
JPEG 분할 화면의 색을 역추출해 누끼로 사용하지 않는다.
소프트웨어 timestamp 근접 매칭은 동일 노출 시각을 보장하지 않으므로 실제 시각 차이와
움직임을 검사하고, 가림/흔들림/불완전한 mask가 있는 시점은 제외한다. 억지로 4장을
채우지 않는다. API 규격에 맞춰 투명 PNG 또는 동일한 단색 배경으로 내보내고, 원본의
색상·비율·손잡이/구멍 등 실제 구조를 보존한다.
