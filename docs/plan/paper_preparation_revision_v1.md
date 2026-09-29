# 논문 준비 수정·평가·재학습 계획 v1

작성: 2026-09-28. 범위: Isaac Gym의 A1 단일 관절 출력 저하, 시뮬레이션 논문. 현재 TF43000/JT71500와 P5 결과를 보존하고 새 프로토콜을 분리한다. 이 문서는 계획이며 학습·평가 실행 결과가 아니다. 이번 작업에서는 구현 및 checkpoint를 변경하지 않았다.

## 1. 논문에서 검증할 주장과 현재 약점

현재 입증한 것: 관측 이력 기반 Student가 지정한 출력 저하 평가에서 높은 생존율을 유지한다. 하지만 이것만으로 공동학습의 우월성, 빠른 고장 식별, 지속적인 추종 회복을 입증하지는 못했다.

논문의 중심 질문 후보:

> 숨은 단일 관절 출력 저하에서 Teacher–Student 공동학습은 동일 정보·예산의 분리학습 및 고장 무작위화 정책보다 지속적인 명령 추종 회복을 개선하는가? 개선되지 않는 조건은 무엇이며, 원인에 맞춘 최소 변경으로 줄일 수 있는가?

단순히 Saving the Limping의 공동학습과 ADAPT의 출력 저하 모델을 조합한 사실만으로 새로운 방법의 기여가 충분하다고 전제하지 않는다. 비교와 원인 분석에서 실제 차별점이 확인되어야 한다. 시뮬레이션만 하는 범위는 유지하며, 실물 검증을 했다는 표현은 사용하지 않는다.

| 우선순위 | 현재 빈약한 부분 | 필요한 근거 |
|---|---|---|
| P0 | evaluator의 명령·관측 일관성 | 실제 policy 입력과 명령 trace assertion |
| P0 | 초기 물리상태의 완전한 짝맞춤 미기록 | 모델과 독립적인 scenario manifest, state/DR snapshot |
| P0 | 평가 지형 column을 family로 읽을 수 없음 | 지형 생성 시 family·물리 난도·mesh hash 기록 |
| P0 | 무유효구간 RMSE=0, action 0.98을 saturation으로 표현 | 결측 처리·분모 공개·물리적 토크 포화와 분리 |
| P1 | Teacher와만 비교, deployable baseline 부족 | 정상학습·고장학습 단일정책·분리학습 Student |
| P1 | 회복 판정이 1초 한 번이면 끝 | 회복시간·추종 유지율·재실패·종료 함께 보고 |
| P1 | 좁은 속도·고장시점 평가 | 연속 무작위 onset, 속도·회전 명령 분해 |
| P1 | 한 학습 정책의 성능 | 주요 비교 최소 3개 독립 학습 seed, 분산 보고 |
| P2 | 취약 셀의 인과 원인 미확인 | 고장 전후 관측/latent/행동/속도/토크 시계열 |
| P2 | 기존 결합을 넘어선 기여 미확정 | 원인 가설 하나와 통제된 ablation |

이미 checkpoint 선택과 분석에 사용한 P5 seed 1–3은 development 결과로 취급한다. 새로운 최종 test scenario를 동결하고 설정·checkpoint 선택에 쓰지 않는다.

## 2. 현재 실제 설정 — 확인된 사실

직접 근거는 [JT 저장 설정](../../logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_/resolved_config.json), [onset 환경](../../legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint.py), [P5 protocol](../../legged_gym/evaluation/p5_paired_protocol.json), [지형 생성 코드](../../legged_gym/utils/terrain.py)다. 논문에서 명시한 값과 프로젝트 설정을 혼동하지 않는다.

### 2.1 고장

- 학습은 이미 2~10초 사이의 **모든 정책 step 후보에서 균등 무작위 onset**이다. 50 Hz이므로 실제 후보는 0.02초 격자다.
- 2/5/10초는 현재 평가의 대표 시점이다. 학습이 세 시점에만 고장을 내는 것이 아니다.
- 학습 열화율은 `{0,.2,.4,.6,.8,1}`에서 선택한다. 연속 d 분포가 아니다.
- 학습 episode는 20초이므로 late onset에서는 고장 후 경험 길이가 10초, early onset에서는 18초 정도다. 평가의 고장 후 20초와 구분한다.
- 정상 d=0 episode와 d=1 완전 상실을 포함한다.

### 2.2 명령

- 학습은 전후·좌우 속도 각각 [-1,1] m/s, 목표 heading [-3.14,3.14] rad, 10초마다 재표본화한다.
- heading 오차로 yaw rate를 생성하는 모드다. 고정 직진만 10초 수행하는 설정이 아니다.
- 현재 평가가 의도하는 명령은 vx=0.5 m/s, vy=0, yaw rate=0이다. 아래 P0의 명령 관측 일관성 검증은 별도 필요하다.
- 현재 Actor는 세계좌표 목표 궤적을 직접 입력받지 않는다. 저수준 속도 명령 추종 정책이다.

### 2.3 지형·무작위화

학습은 trimesh, 8×8 m patch 200개(난도 10행×유형 20열), curriculum 사용이다. 난도 d_terrain은 행 인덱스/10이므로 0~0.9다.

| 학습 지형 | 설정 비중 | 실제 generator 의미 |
|---|---:|---|
| 매끄러운 경사 | 10% | 양·음 경사; slope = 0.4 × 난도 |
| 거친 경사 | 10% | 경사 위 높이 -0.05~0.05 m의 랜덤 요철 |
| 내려가는 계단 | 35% | 현 코드에서 음의 step_height 분기 |
| 올라가는 계단 | 25% | 현 코드에서 양의 step_height 분기 |
| 불연속 장애물 | 20% | 위치·크기 등이 무작위인 사각 장애물 |

계단 높이 크기 = 0.05 + 0.18×난도 → 약 5~21.2 cm, 폭 31 cm. 장애물 높이 설정 = 0.05 + 0.2×난도 → 약 5~23 cm. 경사 최대 기울기 0.36은 각도 약 19.8도다. 부호와 실제 접촉 조건은 patch 진입 방향에도 영향을 받는다. 설정 비중은 map 구성 비중이며, 실제 학습 transition별 노출 비중과 같다고 보장하지 않는다.

마찰 0.5~1.25, 추가 질량 0~2 kg, motor-strength 0.9~1.1, Kp 배율 0.95~1.05, Kd 배율 0.9~1.1, 관측 noise와 15초 간격 push도 학습에 포함된다.

지형은 환경 생성 시 구성하며 episode마다 200개 mesh 전체를 새로 생성하는 것이 아니다. reset 시 curriculum 난도와 초기상태 등이 바뀐다. 고정된 하나의 학습 map에 대한 과적합 가능성과 새로운 map seed 일반화는 분리해서 평가해야 한다.

**평가 지형의 별도 문제:** 현재 evaluator는 `terrain.curriculum=False`로 바꾼다. generator는 이에 따라 `randomized_terrain()`을 호출하고 각 cell에 유형과 난도 {0.5,0.75,0.9}를 무작위 배정한다. 따라서 현재 `terrain_type`은 column index이며 실제 경사/계단 family를 식별하지 못한다. `terrain_level`도 이 경우 물리 난도의 직접 라벨이 아니다.

## 3. 먼저 할 평가 구현 정리 — P0

### 3.1 명령·관측의 시간 일관성

현재 evaluator는 loop 시작에 `env.commands`를 0.5/0/0으로 덮어쓴다. 그러나 환경 callback의 10초 재표본화는 비활성화하지 않는다. callback 뒤에 reward/observation이 계산되고, Student는 반환된 observation을 다음 step에 재사용한다. Teacher는 loop에서 observation을 다시 생성한다.

**판정:** 소스 경로상 랜덤 명령이 Student 현재 관측과 history에 들어갈 가능성이 확인된다. 실제 checkpoint 결과의 변화량은 미측정이다. 이를 이미 입증된 큰 성능 저하 원인이라고 단정하지 않는다. 기존 숫자는 legacy 결과로 남기고, 검증·수정 후 새 평가 버전으로 재측정한다.

수정 계획:

1. evaluation 전용 command provider를 단일 진실 원천으로 둔다.
2. episode reset·callback에서도 지정 command가 유지되거나, 평가 중 재표본화가 실행되지 않도록 한다.
3. reward 기준, 현재 관측의 command slice, history 마지막 frame의 command, 외부 metric의 target이 모든 유효 step에서 일치하는지 assert한다.
4. history를 맞추려고 `compute_observations()`를 두 번 호출하지 않는다. 현재 함수는 history를 한 번 shift하는 부작용이 있다.
5. 10초 직전/직후, onset 직전/직후, reset 직후를 포함한 31초 GPU trace로 검증한다.

### 3.2 시나리오 짝맞춤

Scenario를 policy 실행 전에 생성·저장한다. `scenario_id`, terrain family/난도/mesh hash, 시작 위치·자세·관절각·속도, friction/mass/motor/Kp/Kd, 명령 시간표, fault joint/d/onset, perturbation을 기록한다. Teacher와 Student는 같은 파일을 읽는다. 정책이 달라지면서 이후 접촉·상태가 달라지는 것은 정상이다.

난수는 terrain/initial-state/DR/command/fault stream으로 분리한다. 같은 seed만 넣은 것과 초기 물리조건이 같은 것을 구분한다. GPU 시뮬레이터의 bitwise deterministic 재현을 보장한다고 주장하지 않는다.

### 3.3 지표 수정

- 고장 전 종료는 전체 생존/성공에서 실패로 남긴다. post-fault sample이 0인 RMSE는 null로 기록하고 sample coverage를 보고한다. 0 오차로 평균하지 않는다.
- surviving-only RMSE만 주지 않는다. 유효 구간 RMSE + 생존 + 전체 horizon tracking occupancy를 같이 보고한다.
- `abs(action)>=0.98`은 행동 크기 진단으로 이름을 바꾼다. JT clip_actions=100이므로 actual action/torque saturation이 아니다.
- 필요하면 실제 `abs(tau)/tau_limit`의 임계 초과율을 별도 기록한다. degradation 전후 토크·clip 순서를 명시한다.
- 조기 종료 이유, command change, fault 적용 실제 step, contact, world position/yaw를 저장한다.
- 계측 진단 테스트: 정확한 명령 추종, 한 step spike, 49/50/51 step 안정구간, 회복 후 fall, fault 이전 fall, nonzero yaw command, horizon 마지막 step 종료.

## 4. Onset을 0~10초 무작위로 바꿀 것인가

### 권고

**기존 fixed/2·5·10 평가를 유지하고 연속 무작위 평가를 추가한다. 학습을 곧바로 U(0,10) 하나로 대체하지 않는다.** 완전 무작위 표본도 생성 후 저장해 모든 정책에 동일 적용한다.

| 평가 묶음 | 시점 | 의미 |
|---|---|---|
| Fixed | 정확히 0초 | 이미 고장 난 상태에서 시작 |
| Startup | (0,2)초 무작위 | 초기 자세 안정화·history 부족과 고장이 겹침 |
| In-range onset | [2,10]초 무작위 | 충분한 정상 이력 후 고장; 학습 범위 내 |
| Late onset | [10,18]초 무작위, horizon 연장 | 학습 시점 범위 밖의 고장 시점 |
| Legacy anchors | 2/5/10초 | 과거 결과와 비교하는 회귀 확인 |

0~10초 uniform으로 바꾸면 약 20%가 startup fault가 되어 초기 이력 부족 상황을 더 학습한다. 조기 고장 강건성이 좋아질 가능성과 초기 탐색이 어려워져 학습이 느려지거나 보수적인 보행으로 바뀔 가능성이 모두 있다. **기존 2~10 성능이 좋아질지는 보장하지 않는다.** 평가 분포가 달라져 점수가 바뀐 것과 정책 개선을 구분한다.

0초는 uniform의 극히 작은 일부이므로 fixed branch를 따로 둔다. 0초 고장은 첫 action 전에 적용한다. 현재 callback timing을 그대로 두고 0을 설정하면 첫 action 이후 적용될 가능성이 있어 semantics 검증이 필요하다.

학습 ablation은 onset schedule만 바꾼 A:[2,10], B:[0,10]로 시작한다. post-fault step 노출량도 함께 보고한다. 초기고장 혼합 curriculum은 A/B 결과에 근거해 후속으로 정한다. 정상 d=0 branch는 유지한다.

시점 효과와 다른 변화의 교란:

- 학습의 10초 command switch와 최대 onset 10초가 겹칠 수 있다.
- 0.5 m/s에서 onset 2/5/10초는 nominal 위치 1/2.5/5 m다. 지형 위치도 달라질 수 있으므로 시간만 바뀌는 실험은 우선 평지에서 한다.
- 보행 위상(stance/swing), 고장 전 속도·yaw 안정 상태를 기록한다.

## 5. 명령 추종과 궤적 추종을 나누기

### 5.1 논문 핵심은 저수준 속도 명령 추종

먼저 아래 명령을 같은 fault/scenario 조건으로 평가한다. 수치는 **Implementation choice인 시작안**이며 개발 세트 pilot 후 동결한다.

| 묶음 | 예시 | 확인 대상 |
|---|---|---|
| 직진 anchor | vx=0.5, vy=0, yaw=0 | 기존 논문·현재 결과와 연결 |
| 직진 speed sweep | vx=0.2, 0.5, 0.8 m/s | 낮은/중간/높은 명령 속도 |
| 일정 회전 | vx=0.5, yaw=±0.25 rad/s | 양 방향 회전 중 고장 |
| 변화하는 회전 | yaw=A sin(2πft), vx=0.5 | 시간에 따라 바뀌는 명령 추종 |
| 명령 전환 | vx=0.2→0.8 또는 정지→출발 | transient response |
| 별도 stress | 후진·횡이동 | 기본 결과와 분리한 높은 난도 |

회전 반경은 이상적인 평면 비미끄럼 상황에서 R=v/omega다. 0.5/0.25=2 m. 이것은 속도 명령 패턴이며 world-frame 원 경로를 정확히 따라갔다는 증거는 아니다.

학습은 현재 무작위 10초 command만 유지할지, 일정 직진/곡률 명령을 섞을지 먼저 baseline generalization 결과를 본다. 명령 전환 시간과 fault time은 독립적으로 생성한다. fault-only 회복 시험에서는 고장 전후 command를 일정하게 유지한다. command와 fault 동시 변화는 별도 stress로 분리한다.

ADAPT 원문 Appendix A.2는 전진+회전 중심, 최대 전진 1.0 m/s·회전 0.5 rad/s를 명시한다. 현재 우리 학습은 후진·횡이동까지 포함하므로 학습 과제의 난도가 다르다. 같은 비교라면 command 분포를 맞추거나 차이를 분명히 보고한다. [ADAPT 원문](https://arxiv.org/html/2312.17606v1)

### 5.2 세계좌표 경로 추종은 보조 실험

독립적인 p_ref(s), heading_ref(s)와 공통 상위 path follower를 만든 뒤, 모든 locomotion policy에 동일한 follower·gain·속도 한계를 적용한다.

- 직선: y_ref=0.
- 원: 반경 R=2 m 등의 지정 경로, 양 방향.
- 사인 경로: y_ref(x)=A sin(2πx/L). 곡률에 맞춰 명령 속도 제한.
- 경로 오차: e_perp(t)=min_s ||p_xy(t)-p_ref(s)||, yaw error, 진행률, 완주율.
- 필요시 trajectory time tracking은 ||p(t)-p_ref(t)||로 별도 정의한다. path tracking과 다르다.

단순 속도 적분선을 GT 경로라고 부르지 않는다. 완주한 시행만의 작은 RMSE로 성공을 과장하지 않는다. 시간 제한 내 완료율·진행률을 같이 보고한다. 경로 추가만으로 저수준 정책의 새로운 방법 기여가 생기는 것은 아니다.

## 6. 지형 확장보다 먼저 지형 평가를 명확하게

### 단계

1. **평지:** fault adaptation 자체를 분리해서 원인 분석.
2. **기존 family별 평가:** slope/rough slope/stairs up/down/obstacles × 명시적인 낮음·중간·높음 난도.
3. **새 mesh seeds:** 같은 분포의 새 지형 배치, 학습 map과 분리.
4. **미학습 난도 또는 새 family:** 결과가 안정된 뒤 한 축씩 OOD stress로 추가.

각 family는 난도 label뿐 아니라 slope angle, stair height, roughness amplitude, obstacle height/density 등 물리 파라미터를 저장한다. 전체 점수는 먼저 family별 산출 후 동일 가중 macro average로 보고한다. robot episode가 많은 family가 결과를 지배하지 않게 한다.

8 m patch에서 전진 0.5 m/s를 20초 유지하면 10 m, onset 전 이동까지 더하면 15 m 이상을 이동할 수 있다. patch 경계를 넘어 다른 지형으로 들어간 데이터를 처음 지형 이름으로 집계하지 않도록 한다. 긴 corridor/넓은 동일 family arena를 생성하거나, 실제 지나간 terrain map label을 기록한다. 초기 중앙 평탄 플랫폼만 걸었는지도 검사한다.

train/test의 family·난도·mesh generation seed·실제 배치 범위를 명시한다. terrain.curriculum on/off를 곧바로 “같은 지형에서 자동 승급만 껐다”로 해석하지 않는다.

## 7. 회복률의 정당성과 새 지표 구성

현재 1초 속도/yaw band는 **프로젝트가 정한 과업 기준**이다. 선행논문과 동일한 회복 정의라고 주장하지 않는다. 사용자 정의 지표도 과업 목적, 단위, 고정 임계값, 민감도, baseline 비교가 명확하면 사용할 수 있다.

### 권장 기본 지표

1. 고장 후 20초 완주율, 제한된 평균 생존시간.
2. vx/vy/yaw command RMSE 및 평균 실제 속도(직진 논문과 대응), 평균 bias.
3. 명령 추종 유지율: 고장 후 20초 중 허용 오차를 만족하며 살아 있는 시간 비율.
4. 첫 회복 시간/정해진 시간 내 회복 성공률, 이후 재실패율.
5. 관절×열화율별 결과와 심한 고장군(d≥0.8 또는 d=1)의 별도 macro average.

시간 변화 명령에서도 오차는 actual-target이다. yaw 명령이 0이 아니면 |omega_z|가 아니라 |omega_z-omega_cmd|를 사용한다. 명령 전환에 의해 생기는 정상 transient와 fault-induced recovery는 별도 시험으로 나눈다.

보충 정의:

$$
b_i(t)=\mathbf1\{|v_x-v_x^{cmd}|\le\epsilon_x,\ |v_y-v_y^{cmd}|\le\epsilon_y,\ |\omega_z-\omega_z^{cmd}|\le\epsilon_\omega\},
$$

$$
Q_i=\frac{1}{T}\int_0^T\mathbf1\{\text{alive at }t\}\,b_i(t)\,dt,\qquad T=20\ \mathrm s.
$$

기존 비교용 Q는 vx/yaw의 2조건만 사용하고, 새로운 3축 Q는 별도 이름으로 버전 관리한다. 종료 이후는 Q 기여 0으로 두어 조기 종료가 유리해지지 않게 한다. Q는 원래부터 기준을 벗어나지 않은 시행도 자연스럽게 점수를 받는다.

$$
t_{rec,i}=\inf\{u\ge0:\ b_i(t)=1\text{ for all }t\in[u,u+W]\}.
$$

이는 고장 후 첫 안정구간의 시작 시점이다. 고장 직후부터 계속 기준을 만족하면 t_rec=0으로 정의하되, **추종 유지(no excursion)**와 **이탈 후 회복(regained)**을 별도 분류한다. 한 번 회복 후 재실패하면 recovery-only와 sustained success가 다름을 보여준다.

보고 기준 초안:

- 기존 epsilon_x=0.1 m/s, epsilon_omega=0.2 rad/s, W=1 s를 legacy로 유지.
- 개발 세트에서 epsilon_x {0.05,0.1,0.15}, epsilon_omega {0.1,0.2,0.3}, W {0.5,1,2}에 대한 민감도 표/곡선 확인.
- 여러 속도에서는 epsilon_v=max(epsilon_abs, eta*|v_cmd|) 등의 상대+절대 band를 후보로 비교하되 최종 test 전에 고정한다. 0속도에서 분모가 발산하지 않도록 절대 floor 필요.
- 자세 안전 band를 추가하려면 별도 지표로 정의한다. 경사면에서 비영(非零) roll/pitch를 벌점으로 잘못 잡지 않도록 지면 기준을 고려한다.
- 회복 성공한 episode의 회복시간 평균만 주지 않는다. 실패율과 기한별 회복 누적 비율을 같이 준다. 회복 전 fall은 경쟁 실패사건이며 단순 독립 censoring으로 처리하지 않는다.
- 정상 구간부터 추종이 안 된 시행도 총 성공 분모에서는 제외하지 않는다. 안정 pre-fault 조건부 분석은 별도 보조 결과로 보고한다.

지표를 완화해서 새 모델이 좋아 보이게 만드는 일을 피하려면, legacy와 새 지표를 동시에 산출하고 test를 보기 전에 선택을 고정해야 한다.

## 8. 필요한 baseline과 공정 비교

| ID | 모델 | 답하는 질문 | 우선순위 |
|---|---|---|---|
| B0 | 고장 학습 없는 locomotion | 고장 대응 학습 자체가 필요한가 | 필수 |
| B1 | 같은 fault DR로 학습한 단일 정책 | Teacher–Student가 단순 robust 학습보다 유리한가 | 필수 |
| B2 | Teacher/Actor 동결 후 Student 특징 추정 학습 | Actor까지 공동학습하는 것이 필요한가 | 필수 |
| B3 | 현재 JT 방식 | 재현할 내부 기준 | 필수 |
| B4 | privileged Failure Teacher | 완전 정보 기준, deployable 경쟁자가 아님 | 기준점 |
| B5 | ADAPT 계승 또는 원 구현 | 외부 경쟁 방법과 비교 | 논문용 우선 확장 |

B0도 동일 action/observation 계약을 갖게 한다. 기존 WIM235 checkpoint를 그대로 쓰는 비교는 구조 차이를 공개한 practical baseline으로 표시하고, 엄밀한 paired 비교에는 같은 history/Actor 구조를 정상 환경에서 학습한 control을 고려한다. Base Teacher243처럼 privileged 입력이 있는 모델을 무정보 deployable 정상 baseline이라고 부르지 않는다.

B1은 B0와 같은 구조에서 고장 노출만 바꾼다. 현재 관측-only와 history 포함 버전을 모두 하려면 정보량 ablation으로 구분한다. B2는 같은 teacher initialization, 동일 학습 시나리오·transition 예산에서 Actor를 동결한다. 이미 존재하는 frozen baseline 프로필은 2k warmup/8k transition 등 schedule도 달라 순수 단일변수 비교로 바로 쓰지 않는다.

주요 B1/B2/B3는 최소 3개 독립 학습 seed를 목표로 한다. 가능하면 같은 seed별 Teacher 초기 checkpoint를 B2/B3에 공유하고 seed 간에는 Teacher도 독립 학습한다. Teacher 하나에 Student만 3번 학습하면 “고정 Teacher 조건부 반복”이라고 한정한다.

학습 데이터 예산, 업데이트 수, wall time, 모델 파라미터 수, 추론 지연을 공개한다. Teacher 준비·offline 데이터 생성까지 포함한 비용과 Student 추가 비용을 따로 보고한다. 학습 seed·terrain/scenario cluster 단위 bootstrap 등 계층적 집계로 CI를 산출한다. 한 정책에서 수천 병렬 robot을 돌린 것을 수천 독립 학습 반복으로 보지 않는다.

## 9. 논문 지표·외부 비교

| 연구 | 실제 평가 축 | 현재 프로젝트에 가져올 것 | 동일하다고 주장하면 안 되는 것 |
|---|---|---|---|
| Saving the Limping | 명령 0.5 m/s의 고장 후 속도, 생존시간, 지형별 결과 | 평균 속도+RMSE, 제한 평균 생존시간, 고장 전후 trace | joint locking과 토크 loss는 다른 고장 |
| RMA | success, normalized TTF, 이동거리 | 생존/진행의 평가 축, 분리학습 구조 기준 | 단일 관절 d=1 성능 비교로 해석 |
| ADAPT | d=1 포함 관절×열화율 누적 reward; 추가 다중고장 생존시간 분석 | 같은 degradation 수학 모델, 전진+회전 과업, 외부 baseline | reward를 survival/recovery %로 변환 |
| FT-Net | 고장 전후 velocity tracking, 4 m 전진 후 lateral final deviation, stand-still recovery success | 추종 transient, world-frame lateral drift, 회복 정의 구분 | 넘어진 뒤 일어나는 회복과 현재 1초 tracking recovery를 같은 것으로 취급 |

근거: [Saving the Limping](../papers/main/2023-liu-saving-the-limping/paper.md), [RMA](../papers/main/2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md), [ADAPT 원문](https://arxiv.org/html/2312.17606v1), [FT-Net 원문 Table II / §IV](https://dartmouthrobotics.github.io/icra-2025-robots-wild/spotlight-papers/icra-2025-robots-wild-7.pdf).

위 논문 근거는 **Paper Explicit**다. 우리의 1초 band·threshold·Q 정의는 **Implementation choice**다. 논문 지표를 참고했다는 사실과 정확히 같은 수식·종료·분모를 복제했다는 주장을 구분한다.

### ADAPT 비교의 두 단계

1. 원 구현 audit: [공식 GitHub](https://github.com/WentDong/Adapt), 확인일 2026-09-28. 공개 teacher training/data collection/Student training/evaluation 경로가 있다. paper-specific commit과 사용 가능한 checkpoint는 아직 확정하지 않았다.
2. 원래 pipeline은 12개 Teacher, AMP motion prior, trajectory distillation, Transformer Student다. CNN만 Transformer로 바꾼 모델을 ADAPT라고 부르면 안 된다.
3. 원 환경 sanity reproduction 후 동일 평가 harness로 연결한다. robot asset/PD/action scale/관측·지형·명령·fault/종료·horizon을 맞춘다.
4. 원문 구조를 유지한 benchmark와 관측·학습 과제를 맞춘 adaptation을 구분한다. 후자는 “ADAPT-based reimplementation”으로 표시하고 변경점을 표로 공개한다.
5. 첫 공통 benchmark는 평지, 전진 0.5 m/s, same fault manifest, d-grid including1, 20초 horizon이다. 다음으로 전진+회전, stratified terrain을 확대한다.
6. raw training reward가 아닌 공통 survival/velocity/occupancy 지표로 비교한다. ADAPT의 12 Teacher 비용을 누락하지 않는다.

### 관련연구 갱신 후보

- [AcL, 2025](https://arxiv.org/html/2503.21401v1): multi-teacher와 strict imitation 대신 style guidance, 다중관절 고장. 접근 차이를 관련연구에 반영할 후보이며 모든 baseline을 당장 재현할 필요는 없다.
- [Contrastive Forward Prediction RL, CoRL 2025](https://proceedings.mlr.press/v305/fu25b.html): 예측과 실제 상태 차이를 이용한 적응. 현재 latent 정렬 외의 메커니즘과 비교할 후보.
- [Adaptive Gait Timing, 2026](https://arxiv.org/abs/2608.07328): actuator power loss, latent alignment, gait frequency action을 다룬다. 현재는 초록 수준 확인이며 세부 평가 지표/재현은 아직 audit하지 않았다.

이 후보들은 신규성 판단을 갱신하기 위한 목록이다. 해당 논문의 성능 숫자·세부 구현을 미확인 상태로 주장하지 않는다.

## 10. 재학습에서 바꿀 것 — 한 번에 모두 바꾸지 않기

먼저 세 취약 조건의 GPU trace를 얻는다: RL_hip d=1, RL_thigh d=.8, RR_calf d=.8. 비교적 좋은 FR_hip d=1, 정상 d=0도 대조군으로 기록한다.

| 원인 가설 | 관측할 증거 | 단일 변경 후보 | 반증 기준 |
|---|---|---|---|
| 관측/명령 불일치 | command slice·history에 pulse | evaluator 수정만 | 수정 후 차이가 없으면 주원인 아님 |
| latent 추정이 늦음 | 고장 후 특징/행동 변화 지연, health oracle 대조 | history/auxiliary estimator 변경 | 추정 개선에도 tracking 미개선 |
| 정책과 latent 공동변화가 불안정 | seed별 drift, schedule 구간 민감성 | frozen Actor baseline 또는 schedule 변경 | 동일 budget에서 안정 이점 없음 |
| late-stage 모방 제거 영향 | beta=0 이후 성능 추이 | beta floor 0 vs 0.1 | 평균·severe 회복 개선 없거나 정상 성능 저하 |
| severe 조건 노출 부족 | joint×d별 실제 transition 및 조기종료 분포 | severe sampling 비중 증가 | 단순 예산 증가 효과와 구분 안 됨 |
| 보상과 과업 불일치 | 생존 유지하며 지속적인 speed/yaw bias | failure-conditioned tracking objective 후보 | 생존을 잃거나 평지만 개선 |

beta floor0.1은 기존 코드에 있는 실험 설정이며 최적값/논문값이 아니다. beta가0인 후기에는 Student latent가 Actor와 함께 다른 표현으로 이동할 수 있어 Teacher와의 L2가 큰 것만으로 잘못됐다고 판단하면 안 된다. TF latent를 JT Actor에 단순 교체하는 실험도 표현·Actor 불일치가 생길 수 있다.

**먼저 고를 방향:** baseline과 평가 일관성을 확보한 뒤, severe onset에서 지속 추종이 무너지는 원인을 하나 골라 변경한다. onset 범위·command curriculum·지형·reward·network를 한 run에서 동시에 바꾸지 않는다.

## 11. 실행 순서와 완료 조건

| 단계 | 작업 | 산출물/통과 기준 |
|---|---|---|
| 0 | 기존 모델·결과 동결, P0 evaluator 정리 | 새 protocol v2, SHA manifest, 명령/obs/history 검증 통과 |
| 1 | 기존 TF/JT 재평가, 정상 baseline smoke | legacy-vs-v2 차이 설명, physics 정상 GPU traces |
| 2 | 평지 중심 fault/velocity/severity stratification | 문제 조건·원인 가설, 지표 threshold sensitivity |
| 3 | B0/B1/B2/B3 주요 비교 | 같은 예산/정보, Student-only 평가, 최소 3 training seeds |
| 4 | 원인별 변경 1~2개 ablation | 정상 성능과 survival을 함께 보고, severe occupancy 개선 검증 |
| 5 | ADAPT 기반 비교, 새 terrain/command/onset | 조건 일치표, 외부 비교 지표, unseen scenarios |
| 6 | 최종 test 동결 후 실행 | 선택에 미사용한 test, CI·실패사례·한계까지 보고 |
| 7 | 본문 작성 | 문제/방법/실험/결과/한계의 주장과 표가 일치 |

큰 full-factorial을 처음부터 모두 돌리지 않는다. 평지에서 12관절×d-grid를 충분히 분석하고, terrain 확장은 정상/중간/완전상실 등 축소된 balanced design으로 먼저 확인한 뒤 최종 주요 조건을 늘린다. subset 결과를 전체 12관절의 결과처럼 부르지 않는다.

현재 JT 구간의 과거 로그 시간은 약 14시간 규모였으나, 새 baseline/ADAPT/3-seed 비용의 예측치가 아니다. 첫 pilot 처리량으로 GPU-hour 예산을 산정한다. 학습 완료 조건과 validation checkpoint 선택 규칙을 run 전에 고정한다.

## 12. 구현 파일 대응

- `legged_gym/scripts/evaluate_teacher243_failure_onset.py`, `evaluate_teacher243_failure_matrix.py`: command source, observation timing, terminal handling, trace, 새로운 metric.
- `legged_gym/evaluation/`: 새 versioned scenario protocol/manifest, metric 함수, validity 검사.
- `legged_gym/utils/terrain.py`: family·물리 파라미터 provenance 저장, corridor/arena 생성. 기존 train task와 분리.
- `legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint_config.py`: onset/command distribution의 별도 ablation profile.
- `legged_gym/learning/joint_teacher_student_actor_critic.py`: 현재 기본 구조 보존, 변경 가설에 따라 별도 model/profile.
- `legged_gym/scripts/summarize_p5_paired_evaluation.py`: null coverage, per-training-seed 및 family macro 집계, 생존×회복 contingency.
- `legged_gym/tests/`: command consistency, history one-shift-per-step, metric edge cases, deterministic manifest 소비를 검사.

기존 [회복 진단 계획](jt-student-recovery-diagnosis.md), [beta-floor 계획](jt-beta-floor-ablation.md)은 참고하되 CPU smoke를 성능 결과로 사용하지 않는다. 기존 canonical P5 및 원 checkpoint 파일을 덮어쓰지 않는다.

## 13. 아직 결정할 항목과 추가 조사

- 투고 목표/페이지 수/마감에 따른 재현 범위와 GPU 예산. 현재 계획은 시뮬레이션 연구 논문의 실험 설계를 기본 가정한다.
- ADAPT 원형 재현 vs 공통 조건의 ADAPT-based reimplementation. 코드·checkpoint·데이터 audit 후 비용을 보고 정한다.
- 본 논문의 핵심을 공동학습 효과 검증으로 할지, 확인된 severe-onset 문제의 개선 방법으로 할지. baseline 결과를 보기 전에 개선을 기정사실화하지 않는다.
- recovery의 최종 threshold와 sustained success definition. 개발 세트 sensitivity 및 과업 요구로 정하고 최종 test에는 고정한다.
- 새로운 논문들의 exact fault/termination/metric·공식 코드. 관련연구 후보의 제목·초록 확인과 완전 재현 audit는 구분한다.

현재 당장 필요한 결정은 대규모 재학습 실행이 아니라 **평가 v2를 먼저 구축하고, 같은 환경에서 비교할 B0–B3를 확정하는 것**이다.
