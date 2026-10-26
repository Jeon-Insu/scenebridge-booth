# 물리 에이전트 실험

에이전트는 물체의 준비된 물성을 재사용하고 `configure_experiment`로 실험을 구성한 뒤 `control_physics`로 실행한다. 영상·형상을 물리 스텝마다 다시 추론하지 않는다. 힘의 시간 제어와 PhysX 계산만 스텝마다 진행한다. 이미지로 추정한 재질·질량은 실측값이 아니며 사용자 정정이 우선한다.

## 요청 예시

- “이 컵을 다른 컵 쪽으로 0.01 N·s만큼 밀어 충돌시켜 줘. 상대 컵은 고정해.”
- “같은 충격량으로 컵의 질량만 두 배로 바꿔 다시 실행해.”
- “컵에 동쪽으로 0.02N의 힘을 0.2초 동안 주고 2초간 시뮬레이션해.”
- “멈춰.” / “이어서 재생해.” / “초기화하고 다시 재생해.”
- “어떤 물체와 부딪혔는지 실제 결과를 확인해 줘.”

## 도구와 단위

`push_toward(object_id, target_object_id, impulse_ns, seconds, target_motion, table_material)`는 두 물체 사이의 수평 방향을 계산하는 전용 함수다. 첫 실행에서는 명시적인 테이블 물성 추정치를 함께 전달할 수 있다.

`configure_experiment(bodies, seconds, gravity_m_s2, table_material)`는 실험을 일시정지 상태로 준비한다. `bodies`에는 움직일 물체와 모든 충돌 상대의 정확한 `object_id`를 포함한다. 포함하지 않은 관측 물체는 실험에서 숨겨지며 충돌에 참여하지 않는다. 테이블과 바닥은 항상 고정 충돌체다. 테이블 경계의 보이지 않는 벽은 없다.

| 설정 | 의미 |
|---|---|
| `motion` | `dynamic`: 움직이는 강체, `static`: 고정 장애물 |
| `position_m` | 선택적 초기 위치, 테이블 좌표 m |
| `velocity_m_s` | 초기 속도 [동, 북, 위], m/s |
| `toward_object_id` + `impulse_magnitude_ns` 또는 `speed_m_s` | 다른 참가 객체의 초기 좌표로 수평 방향을 계산하여 충격량·속도를 적용 |
| `impulse_ns` | 시작 시 질량중심에 주는 충격량, N·s. 초기 속도에 J/m을 더함 |
| `force_n`, `force_duration_s` | 시작부터 지정 시간 동안 질량중심에 주는 힘, N 및 s |
| `torque_nm` | 같은 작용 시간 동안 주는 월드 축 토크, N·m |
| `angular_velocity_rad_s` | 초기 각속도, rad/s |
| `mass_kg` | 실험용 질량, kg |
| `static_friction`, `dynamic_friction`, `restitution` | 실험용 정지마찰·동마찰·반발계수 |

벡터는 모두 테이블 고정 축이며 X=동쪽, Y=북쪽, Z=위쪽이다. 현재 회전된 테이블 프레임은 지원하지 않는다. 힘·충격량의 임의 작용점 지정은 제공하지 않으며 토크를 별도로 줄 수 있다. 충돌 형상은 상자·원통·구 근사이고 속이 빈 종이컵의 변형·찌그러짐은 계산하지 않는다.

물성 변경은 `properties_scope=experiment`로 격리한다. 원본 물성과 관측 위치를 수정하지 않는다. 반복 구성은 이전 실험의 초기 위치와 생략된 운동 설정을 재사용한다. 다른 종류의 자극으로 바꿀 때 기존 속도·힘·충격량을 없애려면 해당 벡터를 `[0,0,0]`으로 명시한다. 구성에 포함할 모든 물체의 기본 물성은 먼저 준비되어 있어야 한다.

`control_physics(play|pause|reset|return)`로 제어한다. 완료된 실험은 `reset` 후 `play`한다. 정지·실험 준비·재생 사이에 Kit 업데이트를 기다리며, 준비가 실패하면 뒤의 재생 명령은 실행하지 않는다. 화면은 Isaac 적용 확인 후 성공을 표시한다.

`get_experiment_result()`는 마지막 고정 스냅샷의 실제 시뮬레이션 시간, 초기·최종 위치, 자세, 속도, PhysX 접촉 객체 쌍을 반환한다. `net_displacement_m`는 시작점과 끝점 사이의 3차원 직선 변위이며, 전체 이동 경로 길이는 아니다. 객체 간 가까워짐만으로 충돌을 판정하지 않는다. 결과의 `completed=false`이면 진행 중 또는 중간 정지 결과다. 접촉 목록은 한 실험에서 객체 쌍별 첫 접촉을 보존하며 모든 충돌 충격량·횟수를 분석하는 도구는 아니다.

## 검증

- Python: `.venv-mocap/bin/python -m pytest tests/smartphone_mocap -q`
- Isaac: `python.sh tools/validate_interaction_physics.py`
- 기존 낙하: `python.sh tools/validate_viewport_physics.py`
- 확장·실행 승인: `python.sh tools/validate_tabletop_twin_extension.py`

엔진 검증은 실제 앱 프레임을 진행한다. 반복 실험에서 앱 업데이트를 생략하고 물리 스텝만 호출하면 타임라인 전환을 반영하지 못할 수 있다.
