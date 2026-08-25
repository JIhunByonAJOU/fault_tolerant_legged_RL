# BaseEnv privileged teacher 구현 기록

기준일: 2026-08-10

학습 명령, TensorBoard와 RTX 4090 batch benchmark는
[`training-operations.md`](training-operations.md)에 기록한다.

## 범위

task 이름은 `a1_limping_base`다. joint failure, student encoder, history,
adaptation loss와 alpha/beta schedule은 포함하지 않는다. 현재 경로는

```text
42-D deployable observation o ─────────┐
                                      ├─ policy pi → 12 actions
234-D clean simulator GT e → mu → 8-D z
                                      └─ asymmetric critic
```

이며 raw GT를 policy에 직접 concat하지 않는다.

## 구현 위치

- environment/config/schema: `legged_gym/envs/a1_limping/`
- teacher model/PPO/runner: `legged_gym/learning/`
- task registration: `legged_gym/envs/__init__.py`
- local runner selection: `legged_gym/utils/task_registry.py`
- tests: `legged_gym/tests/test_a1_limping_base.py`,
  `legged_gym/tests/test_teacher_actor_critic.py`

외부 `/home/jihun/Capstone2/rsl_rl`은 수정하지 않았다. rollout storage와
PPO 기본 계산은 재사용하되, stock PPO가 privileged observation을 actor에
전달하지 않으므로 local `TeacherPPO`와 `TeacherOnPolicyRunner`가 observation과
GT를 분리해 전달한다.

## Tensor schema

Actor observation은 42-D다.

| slice | 값 | 차원 |
|---|---|---:|
| `[0:12]` | default 기준 joint position | 12 |
| `[12:24]` | joint velocity | 12 |
| `[24:26]` | wrapped roll, pitch | 2 |
| `[26:30]` | binary foot contact | 4 |
| `[30:42]` | previous action | 12 |

Privileged observation은 234-D다.

| slice | 값 | 차원 |
|---|---|---:|
| `[0:1]` | applied friction | 1 |
| `[1:2]` | added base mass | 1 |
| `[2:14]` | motor-strength ratio | 12 |
| `[14:26]` | Kp ratio | 12 |
| `[26:38]` | Kd ratio | 12 |
| `[38:41]` | failure state; BaseEnv에서는 모두 0 | 3 |
| `[41:47]` | clean body linear/angular velocity | 6 |
| `[47:234]` | 17×11 local height map | 187 |

적용 전 metric/ratio 값은 `privileged_obs_raw_buf`에, network normalization을
거친 값은 `privileged_obs_buf`에 보존한다. 현재 friction만 conservative
randomization이 켜져 있다. base-mass randomization은 값을 기록할 수 있게
구현했지만 기본값은 off이며, motor/Kp/Kd ratio는 실제 적용값 1이다.

## Network와 reward

- teacher encoder: `234 → [512,256,128] → 8`, ELU
- actor: `50 → [256,128] → 12`, ELU
- critic: `50 → [512,256,128] → 1`, ELU
- fixed command: `(vx, vy, yaw)=(0.5,0,0)`
- normalized action clip: `[-1,1]`; 이후 0.25 rad position scale 적용
- reward: Saving이 closely follows한다고 밝힌 RMA의 10개 term과 scale을
  계승하고 forward cap만 0.5 m/s로 변경
- penalty term 3–10 curriculum: `k0=0.03`,
  `k_t = k0 ** (0.997 ** iteration)`
- termination: target papers에 scale이 없으므로 `-200` implementation choice.
  WIM dt 적용 후 `-4/fall`이다. `-1000`(`-20/fall`) ablation도 20초 생존을
  개선하지 못해 기본값에는 반영하지 않았다.
- main PPO: RMA supplement에서 `gamma=0.998`, `lambda=0.95`, Adam
  `learning_rate=5e-4`, value coefficient `0.5`, entropy `0`, update round `4`를
  계승. WIM reward control은 기존 workspace PPO 설정 유지
- RTX 4090 기본 throughput profile: `32,768 env × 24 steps`, 8 minibatches
  (`786,432` transitions/update, `98,304` samples/minibatch). 측정 기준 약
  `327–340k transitions/s`, active GPU utilization `78.2%`, VRAM `12.5 GB`이며
  main과 WIM control 양쪽에 동일 적용. 49k–57k env는 VRAM만 늘고 느려져 제외

`Z Acceleration` 항목은 RMA 본문에 인쇄된 식 `-||v_z^t||²`를 구현했다.
명칭과 식의 불일치는 source ambiguity로 유지한다. reward scale에는 현재
WIM의 control-dt multiplication이 적용되므로 이는 target 직접값이 아니라
Isaac Gym 이식 선택이다.

## 검증 결과

다음 검증을 `legged_ws`, RTX 4090, Isaac Gym GPU pipeline에서 통과했다.

1. CPU model test 3개: shape, teacher gradient, GT sensitivity
2. 64-env integration: 42/234-D, finite tensor, fixed command, neutral failure
   state, previous-action timing, RMA reward registration, 7 physics steps
3. 두 개의 독립된 16-env simulation에서 fixed-seed 10-step trace bitwise 비교
4. 64-env PPO 10 iterations: 15,360 transitions, checkpoint/config 저장, NaN 없음
5. 4,096-env PPO 1 iteration: 98,304 transitions, 약 192k transitions/s,
   checkpoint/config 저장
6. 기존 `a1` stock runner 64-env 1 iteration 회귀 통과
7. 10-iteration checkpoint reload 후 64-env×20초 teacher inference 완료. 이
   checkpoint의 평균 전진 속도는 약 -0.003 m/s로, wiring만 검증됐을 뿐
   locomotion 학습 성공은 아님
8. action clip 수정 후 4,096-env에서 총 150M-transition main run 완료.
   256-env×20초 evaluation은 최종 survival `1.56%`, 평균/중앙 전진 속도
   `0.074/0.082 m/s`로 locomotion gate 미통과
9. periodic checkpoint의 파일명 iteration과 직렬화된 `iter` 불일치를 발견해
   local teacher runner가 파일명 기준으로 legacy checkpoint를 보정하고, 새
   checkpoint에는 실제 global iteration을 저장하도록 수정. `model_755.pt`의
   내부 `iter=755`를 직접 확인

실행 명령:

```bash
/home/jihun/Capstone2/miniconda3/bin/conda run -n legged_ws \
  python legged_gym/tests/test_a1_limping_base.py \
  --headless --sim_device=cuda:0 --num_envs=64

/home/jihun/Capstone2/miniconda3/bin/conda run -n legged_ws \
  python legged_gym/scripts/train.py \
  --task=a1_limping_base --headless --sim_device=cuda:0 \
  --num_envs=64 --max_iterations=10

/home/jihun/Capstone2/miniconda3/bin/conda run -n legged_ws \
  python legged_gym/scripts/evaluate_teacher.py \
  --task=a1_limping_base --headless --sim_device=cuda:0 \
  --num_envs=64 --load_run=<run-directory> --checkpoint=<iteration>
```

## 아직 완료되지 않은 gate

- no-failure 20초 survival `>=90%`, median vx `>=0.35 m/s` 달성
- smooth slope, rough slope, discrete obstacle 단계화
- base mass, motor strength, Kp/Kd conservative DR 활성화
- GT zero/shuffle evaluation을 통한 teacher의 privileged-information 사용 확인
- 최소 3 seed 재현

따라서 현재 상태는 **teacher-only vertical slice 구현, smoke 및 150M 학습
진단 완료**이며, BaseEnv teacher locomotion 성공 판정은 아직 아니다.

## 25M 초기 실험 기록

동일 seed 1, plane, conservative friction DR에서 확인했다.

| arm | reset | termination 선택 | survival | 평균 vx |
|---|---|---|---:|---:|
| RMA-derived | WIM wide reset | 없음 | 0% | 0.245 m/s |
| WIM reward control | WIM wide reset | WIM default | 99.2% | 0.138 m/s |
| RMA-derived | stationary/near-nominal | 없음 | 0% | 0.468 m/s |
| RMA-derived | stationary/near-nominal | 낙상당 -4 | 0% | 0.360 m/s |

위 RMA-derived 25M run들은 WIM의 `clip_actions=100`을 잘못 상속한 pre-fix
결과다. RMA-PPO 계승 run에서 action magnitude가 폭증하며 이 누락을 확인했고,
paper/계획대로 `clip_actions=1`로 수정한 뒤 main gate를 다시 수행한다.

따라서 model/GT/PPO 경로는 WIM control에서 검증됐고, RMA-derived arm은
목표 속도는 학습하지만 조기 낙상을 회피하지 못했다. target paper에 없는
termination scale을 위 implementation choice로 추가한 후 다음 gate를 수행한다.

## action-clip 수정 후 150M main gate

run: `Aug07_17-38-55_base_150m_seed1_actionclip`; seed 1, 4,096 env,
plane, friction DR `[0.5,1.25]`, action clip 1, 낙상당 -4.

| checkpoint | 누적 transition(근사) | survival | 평균 vx | 중앙 vx | 해석 |
|---:|---:|---:|---:|---:|---|
| 500 | 49M | 51.17% | 0.009 | 0.003 | 생존 일부 확보, 사실상 정지 |
| 750 | 74M | 0% | -0.060 | -0.059 | 붕괴 |
| 1000 | 98M | 0% | 0.227 | 0.226 | 전진하지만 조기 낙상 |
| 1250 | 123M | 2.34% | 0.125 | 0.133 | gate 미달 |
| 1526 | 150M | 1.56% | 0.074 | 0.082 | 최종 gate 미달 |

TensorBoard 학습 중간 로그는 다음과 같다. episode length는 control step 수이며
1 step은 0.02초다. transition은 iteration당 `4096*24=98,304`로 환산했다.

| iteration | 누적 transition | 평균 episode step | 평균 reward | penalty scale | value loss | noise std | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 255 | 25.1M | 12.8 | -1.70 | 0.189 | 0.78 | 0.768 | 231k |
| 300 | 29.5M | 300.1 | -13.88 | 0.241 | 1.36 | 0.828 | 219k |
| 500 | 49.2M | 1002.0 | -19.30 | 0.458 | 2.31 | 0.851 | 228k |
| 750 | 73.7M | 67.0 | -27.27 | 0.692 | 7.92 | 0.865 | 221k |
| 1000 | 98.3M | 39.2 | -15.69 | 0.841 | 13.04 | 0.903 | 208k |
| 1250 | 122.9M | 54.7 | -110.54 | 0.921 | 1543.66 | 0.968 | 211k |
| 1500 | 147.5M | 46.3 | -140.72 | 0.962 | 3914.07 | 1.070 | 224k |
| 1525 | 149.9M | 33.9 | -119.47 | 0.965 | 3110.59 | 1.080 | 212k |

49M 부근에는 episode를 거의 끝까지 채우지만 deterministic evaluation에서는
절반만 생존하고 중앙 vx가 0.003 m/s인 정지 정책이다. 이후 penalty scale이
0.69 이상으로 올라가면서 episode length가 급락하고, 123M 부근부터 value
loss가 세 자릿수에서 수천 단위로 폭증한다. 같은 시점의 lateral/rotation
episode term도 iteration 500의 `-0.160`에서 1250의 `-4.822`, 1500의
`-6.791`로 커졌다. 따라서 단순히 학습량이 부족한 곡선으로 보기는 어렵다.

비교군 `Aug07_17-21-50_base_wim_25m_seed1`은 iteration 254에서 평균 episode
length 959.6 step, evaluation survival 99.2%, 평균 vx 0.138 m/s였다. 이 run은
action clip/reset 수정 전 control이라 최종 후보로 사용하지 않고, 현재 설정으로
다시 학습해야 한다.

`-20/fall` ablation은 생존이 가장 높았던 `model_500`에서 25M을 더 학습했다.
checkpoint iteration 보정 후 만든 run
`Aug07_17-54-02_base_term20_25m_from500_iterfix_seed1`의 `model_755`도
survival 0%, 평균/중앙 vx `-0.009/-0.007 m/s`여서 개선되지 않았다. 따라서
기본 termination은 `-4/fall`로 복구했다.

관측/GT/teacher/PPO 경로는 finite하고 WIM reward control이 25M에서 survival
99.2%를 보였으므로, 다음 교정 대상은 RMA reward의 Raisim→Isaac Gym 단위
이식과 penalty curriculum이다. 이 교정 전에는 FailureEnv를 구현하지 않는다.

### Action saturation 추가 진단

서로 다른 `model_500`, `model_550`, `model_600`, `model_650`이 deterministic
evaluation에서 완전히 같은 결과를 내어 raw actor output을 추가 측정했다.

| checkpoint | mean `abs(raw action)` | `abs(action) >= 1` 비율 | survival |
|---:|---:|---:|---:|
| 500 | 1234.09 | 100% | 51.17% |
| 600 | 486.57 | 100% | 51.17% |
| 750 | 126.24 | 99.999% | 0% |

checkpoint의 SHA-256과 teacher/actor/critic parameter는 서로 다르므로 load 오류가
아니다. network output이 폭주한 뒤 environment의 `[-1,1]` clip에서 같은 physical
action으로 포화된 것이다. 따라서 150M 실패를 reward만의 문제로 단정할 수 없고,
RMA PPO/critic/teacher encoder 결합이 현재 rsl_rl/Isaac Gym에서 안정적인지도 함께
분리해야 한다. evaluator는 이후 모든 run에서 raw action 평균과 saturation rate를
출력한다.

### 2026-08-09 교정 상태

위 진단에 따른 첫 교정에서는 teacher actor를 tanh-squashed Gaussian으로 바꾸고
bounded action을 rollout storage에 저장해 PPO log-prob을 계산했다. observation의 previous-action
1-step 추가 지연도 제거했다. 보상/penalty curriculum은 원인 분리를 위해 아직
변경하지 않았다. 다음 main gate는 기존 checkpoint resume이 아니라 seed 1의 새
run으로 수행한다.

후속 8k run에서 bounded action을 `atanh`로 복원하는 PPO가 iteration 57 이후
ratio 폭발/NaN으로 종료됐다. 최종 교정에서는 rollout storage에 pre-tanh Gaussian
sample을 보관하고 simulator에만 tanh 결과를 전달한다. PPO는 raw 좌표에서 ratio를
계산하며, adaptive desired KL `0.01`, hard KL stop `0.05`, finite guard를 사용한다.
8,192-env 120-iteration 회귀에서 NaN 없이 이전 실패 지점을 통과했고 평균 전진
속도 `0.457 m/s`를 얻었으나 20초 survival은 0%였다. 따라서 PPO 안정화 gate는
통과했지만 BaseEnv locomotion/survival gate는 계속 진행 중이다.

후속 분석에서 privileged observation 참조가 `env.step()` 중 in-place 갱신되어
`e_t` action과 `e_{t+1}` storage가 짝지어진 것을 확인했다. action 시점에 actor 및
privileged observation을 clone하도록 수정한 뒤, 8,192-env/seed-1 100-iteration
회귀에서 1,500개 deterministic 평가 모두 20초 생존했고 평균/중앙 속도는
`0.554/0.554 m/s`였다. BaseEnv teacher의 plane locomotion gate는 이 checkpoint로
처음 통과했다.

## `a1_limping_base_v2` — command-conditioned MDP

V1과 checkpoint는 보존하고 별도 task로 추가했다. 구조는 다음과 같다.

```text
42-D Saving/RMA sensor state + 3-D [vx, vy, yaw-rate] command ─┐
                                                               ├─ policy → 12 actions
234-D simulator GT → Saving teacher encoder → 8-D latent ──────┘
```

기본값은 `8,192 env × 24 steps × 3,000 iterations = 589,824,000`
transitions다. 평지·failure 없음은 V1과 같고, actor/critic 입력만 `50→53`으로
확장했다. command는 privileged GT가 아니라 student도 사용할 수 있는 외부 입력이다.

| 항목 | v2 선택 | 근거/불확실성 |
|---|---|---|
| command | 50%는 Saving `(0.5,0,0)`, 50%는 `vx∈[0,0.8]`, `vy∈[-0.3,0.3]`, `yaw∈[-0.5,0.5]`; 10초마다 변경 | **Implementation choice**. fixed-command regression과 3-D conditioning을 동시에 확보; WIM code default `[-1,1]`보다 보수적으로 축소 |
| task reward | WIM exponential XY/yaw tracking, `sigma=0.25`, scale `10/5` | 식과 2:1 비율은 **Paper Explicit + Code Experiment — WIM**. RMA forward의 최대 pre-dt reward `20×0.5=10`과 단위를 맞춘 scale은 **Implementation choice** |
| fixed stability | WIM/A1 `lin_vel_z`, `ang_vel_xy`, torque, acceleration, action-rate, airtime, collision, joint-limit | **Code Experiment — official legged_gym**. WIM 논문 reward 표와 공개 코드 coefficient 일부가 달라 code path를 우선 |
| natural constraints | RMA work, ground-impact, torque smoothness, foot-slip과 공개 원계수 | **Inherited — Saving→RMA**. task scale을 RMA와 맞췄으므로 별도 0.1배 이식 없음 |
| base height | target `0.25 m`, scale `-0.6` | Saving/WIM 비영점 계수는 **Unspecified**; Saving이 인용한 Rapid Locomotion의 값 사용 |
| PPO | 현재 안정화된 RMA-derived teacher PPO 유지 | Saving PPO는 **Unspecified**. MDP 효과를 분리하기 위해 동시에 WIM PPO로 바꾸지 않음 |

W&B에는 매 iteration command RMSE, vertical-velocity RMS, base height, impact,
slip, diagonal/four-feet/airborne contact rate가 기록된다. episode가 끝나면 survival,
world displacement, path efficiency, net yaw와 base-height peak-to-peak도 기록한다.
학습 시작 시 random episode length 때문에 생긴 짧은 timeout은 20초 survival
분모에서 제외한다.

검증 상태:

1. compile 및 teacher/PPO unit test 7개 통과
2. v1+v2 각각 64-env Isaac Gym integration 통과
3. v2 64-env PPO 2-iteration smoke 통과; actor/critic 첫 입력 `53`, finite
   reward/PPO 확인
4. 최종 설정으로 8,192-env/100-iteration smoke 완료; 19,660,800 transitions,
   NaN·finite skip 없음, PPO max KL `0.0269 < 0.05`

100-iteration checkpoint를 256개 env, 고정 command `(0.5, 0, 0)`, deterministic
action으로 20초 평가한 결과는 다음과 같다.

| 지표 | 결과 | 판정 |
|---|---:|---|
| 20초 survival | 100% | 통과 |
| 전진 속도 / vx RMSE | 0.171 m/s / 0.340 m/s | 방향 학습은 시작했지만 부족 |
| path efficiency / net yaw | 0.981 / 0.149 rad | 직진성 양호 |
| vertical velocity RMS / base-height p-p | 0.450 m/s / 0.253 m | 불합격: bounce가 큼 |
| four-feet / diagonal contact | 71.6% / 0.007% | 불합격: 자연 gait가 아니라 stomping에 가까움 |

WIM 원 scale `1/0.5`는 같은 smoke에서 deterministic stationary policy로
수렴했다. 최종 `10/5`와 50% fixed Saving command는 정지해를 벗어나 직진 학습을
시작하게 했지만, 둘 다 논문 명시값이 아니라 smoke 근거의 implementation choice다.

현재 결론은 **MDP/PPO wiring은 통과, gait 수렴은 미확인**이다. 따라서 terrain이나
failure를 붙일 단계는 아니며, 평지 본 학습에서 checkpoint 250/500/1,000을 먼저
평가한다. 속도 RMSE가 감소하면서 vertical/contact 지표도 함께 좋아지면 3,000까지
진행하고, 속도만 좋아지고 stomping이 남으면 M2 selective trot prior를 비교한다.
