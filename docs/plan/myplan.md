여기에는 메인 논문이랑 주변 정보들을 찾아서 실행 계획을 목적성에 맞게 작성하는 곳.

# Saving the Limping first-pass 재현 실행 계획

## 사용자 결정 기록 — 2026-08-07

다음 범위와 기본 방향을 확정했다.

1. **simulation-only**: Isaac Gym 내부 실험만 수행한다. 실제 A1 배포, real-world softlock, real-world hardlock, 3D-printed lock mechanism은 구현·시험 범위에서 제외한다. 단, 논문 재현을 위한 simulator actor DOF-limit joint lock은 핵심 환경 mechanics이므로 유지한다.
2. failure tolerance의 음수 정규분포 해석은 first pass에서 half-normal과 bounds clip을 사용하고 raw sample·변환값·source label을 모두 기록한다. 사용자가 수치를 위임했으므로 아래 표 C의 recommendation을 기본값으로 채택하되 sensitivity를 필수로 유지한다.
3. failure time은 아래 표 C의 `U(2,8) s` recommendation으로 시작하고 fixed-time evaluation과 bounds ablation을 유지한다.
4. 30-D observation은 `Saving → RMA` 경로의 A1 sensor 구성에 맞춘다.
5. reward는 `Saving`의 explicit 목적과 `0.5 m/s`를 우선하며, 공개되지 않은 식·scale은 논문이 “closely following [7], [9]”라고 밝힌 계보에서 가져온다. target 직접값이 아닌 항목은 계속 `Inherited`/`Implementation choice`로 기록한다.
6. DR은 conservative range로 먼저 수렴시키고, wide range는 후속 ablation으로만 확장한다.
7. 첫 compute target은 단일 RTX 4090 24 GB다. 25M/150M gate에서 memory 또는 throughput이 부족할 때 H100으로 확장한다.
8. **adaptation gradient**: $L_{\mathrm{adapt}}$의 teacher latent는 detach하고 student encoder만 이 loss로 갱신한다. 단, Saving joint-training 중 teacher encoder는 $\alpha<1$인 동안 PPO reward 경로로는 계속 갱신될 수 있다. 즉 “teacher 고정”은 adaptation target에 대한 말이지 전체 optimizer에서 영구 freeze한다는 뜻은 아니다.
9. **재현 성공**: 논문 숫자의 exact match는 요구하지 않는다. joint training이 정상적으로 되고, 공통 failure evaluation에서 FailureEnv가 BaseEnv보다 높은 성능을 보이며 논문과 같은 방향성을 보이면 성공으로 판단한다. 논문 숫자는 reference overlay로만 남긴다.
10. **learning stack**: 외부 `/home/jihun/Capstone2/rsl_rl`은 수정하지 않고 이 repository 안에 custom model/PPO/runner/storage를 둔다. 외부 package는 upstream-compatible dependency로 고정한다.
11. **critic input**은 Saving 본문에 미기재다. 가장 가까운 [9] Rapid Locomotion 공식 구현은 privileged encoder latent와 critic observation을 함께 value network에 넣는다. first pass 권장은 같은 계보의 asymmetric critic이며, 아래 설명을 본 뒤 최종 선택할 수 있다.
12. **학습 순서**: BaseEnv privileged teacher가 모든 현재 ground truth를 사용해 정상 보행을 학습하는지 먼저 확인한다. 그다음 deterministic joint-lock test와 FailureEnv privileged teacher를 통과시키고, 마지막에 BaseEnv/FailureEnv joint training을 수행한다.

실행 환경 구성과 검증 결과는 [`legged_ws-environment.md`](legged_ws-environment.md)에 고정했다.

## 목표와 재현 범위

현재 `leggedrobotics/legged_gym` HEAD `8fa29acc6fd1910c3d9659eef6310bdd301cde0a`에서 Unitree A1이 **episode 도중 임의의 한 관절이 잠겨도 이전보다 오래 서고 전진하는가**를 검증한다. 논문 [Saving the Limping](../papers/main/2023-liu-saving-the-limping/paper.md)의 명시 사실을 우선하고, 빠진 수치는 [RMA [7]](../papers/main/2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md), [Rapid Locomotion [9]](../papers/ref/2022-margolis-rapid-locomotion-via-reinforcement-learning/paper.md), [Rudin et al. [21]](../papers/main/2021-rudin-learning-to-walk-in-minutes-using-massively-para/paper.md)와 공식 코드에서 가져오되 항상 “계승값/구현 선택”으로 기록한다.

완료 선언은 “논문 수치 완전 재현”이 아니라 다음 세 조건으로 제한한다.

1. baseline A1 locomotion과 joint-lock FailureEnv가 결정론적으로 동작한다.
2. teacher → history student → joint training 전체 경로가 학습되고 student-only checkpoint가 평가된다.
3. 동일 failure seed matrix에서 FailureEnv student가 BaseEnv student보다 survival/velocity degradation을 개선하는지 통계로 판단한다.

공식 target code 감사와 세부 provenance는 [구현 근거 조사](../papers/analysis/saving-the-limping-implementation-search.md)에 있다. **2026-08-06 현재 공식 공개 학습 코드는 발견되지 않았다.** 저자 프로젝트 Git 전체 이력에는 웹 페이지와 영상만 있고 저자의 Rapid Locomotion 포크에도 lock code가 없다.

## 근거 규칙

- `Paper Explicit`: 해당 **근거 사실만** 이름을 적은 논문이 직접 명시했다. 같은 행의 구현 선택 전체가 논문값이라는 뜻이 아니다.
- `Inherited`: target이 인용한 구현 선조 논문에서 가져온 사실이다. 반드시 `Saving → RMA`처럼 경로를 쓴다.
- `Code Experiment`: 논문 공식 저장소의 paper-specific 실행 경로에서 확인한 값이다.
- `Code Default`: 공식 또는 현재 코드의 기본값이지만 target 실험에서 사용했는지는 확인되지 않았다.
- `Derived`: explicit 근거로부터 계산한 값이다. 계산식도 함께 쓴다.
- `Unspecified/Ambiguous`: 공개 근거에 값이 없거나 해석이 둘 이상이다.
- `Implementation choice`: 위 근거와 별도의 열이다. 실제 first pass에 넣을 값과 선택 주체를 적으며, `Agent recommendation`이면 제가 만든 값이다.

따라서 한 행에 `Paper Explicit — Saving`과 `Agent recommendation`이 함께 보인다면, **논문이 명시한 상위 조건은 유지하지만 논문이 비워 둔 하위 수치를 제가 채웠다**는 뜻이다. 혼동을 줄이기 위해 아래 표에서는 가능한 한 이런 경우를 별도 파라미터 행으로 분리한다.

## 출처 지도

아래 표의 이름을 이후 파라미터 표에서 반복해 사용한다. `Saving` 이외의 값은 모두 target 논문의 값이 아니라 계승 후보 또는 구현 참고값이다.

| 약칭 | 논문·코드 | 이 계획에서의 역할 | 직접 확인 위치 |
|---|---|---|---|
| `Saving` | [Saving the Limping](../papers/main/2023-liu-saving-the-limping/paper.md) | 재현 대상이자 `Paper Explicit`의 유일한 기준 | Sec. III-A–C, Sec. IV-A–E, pp. 3–7 |
| `RMA` | [Rapid Motor Adaptation for Legged Robots](../papers/main/2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md) | `Saving`이 observation, reward, student CNN을 계승한 참고문헌 [7] | Sec. III-A–B, Sec. IV-A–B 및 supplement |
| `Rapid` | [Rapid Locomotion via Reinforcement Learning](../papers/ref/2022-margolis-rapid-locomotion-via-reinforcement-learning/paper.md) | `Saving` 참고문헌 [9]; DR와 teacher/student 구현 비교 | [공식 저장소, commit `f5143ef`](https://github.com/Improbable-AI/rapid-locomotion-rl/tree/f5143ef940e934849c00284e34caf164d6ce7b6e) |
| `WIM` | [Learning to Walk in Minutes](../papers/main/2021-rudin-learning-to-walk-in-minutes-using-massively-para/paper.md) | `Saving` 참고문헌 [21]이자 현재 플랫폼의 원 논문 | 현재 [legged_gym 저장소](https://github.com/leggedrobotics/legged_gym), local HEAD `8fa29acc` |
| `rsl_rl` | WIM이 사용하는 PPO 구현 | 현재 runner/storage가 지원하는 범위를 판단 | [공식 `v1.0.2`, commit `2ad79cf`](https://github.com/leggedrobotics/rsl_rl/tree/2ad79cf0caa85b91721abfe358105f869a784121) |

## WIM 코드에 무엇을 붙이는가

현재 코드는 `train.py → task_registry → A1RoughCfg + LeggedRobot → rsl_rl.OnPolicyRunner` 순서다. `Saving`을 넣으면 다음처럼 바뀐다.

```text
train.py
  └─ task_registry
      ├─ a1_limping_base    ─┐
      └─ a1_limping_failure ─┴─ A1LimpingEnv
                                  ├─ 42-D deployable observation
                                  ├─ 50-frame history
                                  ├─ privileged environment vector
                                  └─ joint-lock scheduler → Isaac Gym DOF limits
                                           │
                                           ▼
                                LimpingOnPolicyRunner
                                  ├─ teacher encoder μ
                                  ├─ student encoder φ
                                  ├─ fused policy π
                                  └─ PPO + adaptation loss
```

학습 gate는 `BaseEnv privileged teacher → FailureEnv privileged teacher → student/joint training` 순서로 둔다. joint-lock mechanics 자체의 deterministic unit test는 FailureEnv teacher 전에 수행하지만, failure 학습을 BaseEnv teacher보다 먼저 시작하지 않는다. 먼저 failure가 없는 조건에서 42차원 관측, privileged ground truth, teacher encoder, policy, critic, reward와 PPO가 정상적으로 locomotion을 학습하는지 확인해야 이후 실패 원인을 분리할 수 있다.

## 파라미터 표 A — 플랫폼, 시간, 명령

| 파라미터 | 짧은 설명 | 확인된 사실·값 | 근거 분류·정확한 위치 | 이번 구현 선택 | 선택 근거 |
|---|---|---|---|---|---|
| 로봇·시뮬레이터 | 학습 대상 robot asset과 physics backend | Unitree A1, Isaac Gym 및 IsaacGymEnvs | `Paper Explicit — Saving`, Sec. IV-A, pp. 4–5 | 현재 A1 URDF와 Isaac Gym Preview 3 사용 | `Implementation choice`; 현재 [WIM A1 Code Default](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:63). target asset과 완전히 같다는 근거는 없음 |
| 본 학습 병렬 환경 수 | GPU 한 장에서 동시에 돌릴 environment 개수 | GPU당 4,096개, NVIDIA A6000 두 장 | `Paper Explicit — Saving`, Sec. IV-A, p. 4 | GPU당 4,096개 | 논문 그대로; 별도 선택 없음 |
| smoke-test 환경 수 | 구현 오류를 빠르게 찾기 위한 소규모 환경 수 | 논문에 없음 | `Unspecified — Saving` | 64개 | `Implementation choice — Agent recommendation` |
| simulation rate | physics integration 빈도 | $f_{\mathrm{sim}}=200\,\mathrm{Hz}$ | `Paper Explicit — Saving`, Sec. IV-A, p. 4 | $\Delta t_{\mathrm{sim}}=1/200=0.005\,\mathrm{s}$ | `Derived`; [WIM sim default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:183)와도 일치 |
| control rate | policy action을 새로 계산하는 빈도 | $f_{\mathrm{ctrl}}=50\,\mathrm{Hz}$ | `Paper Explicit — Saving`, Sec. IV-A, p. 4 | decimation $=200/50=4$ | `Derived`; [WIM A1 Code Default](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:53)와 일치 |
| 목표 전진 속도 | reward가 추종할 commanded forward speed | $v_x^\star=0.5\,\mathrm{m/s}$ | `Paper Explicit — Saving`, Sec. III-B, p. 4 | $v_x^\star=0.5\,\mathrm{m/s}$ | 논문 그대로; 별도 선택 없음 |
| command 구성 | policy에 줄 선속도·횡속도·yaw 명령의 범위 | 분포와 나머지 축 값은 없음 | `Unspecified — Saving` | $\mathbf c_t=[0.5,0,0]$으로 고정 | `Implementation choice — Agent recommendation`; 변수를 줄이는 first pass |
| 학습 episode 길이 | reset 전 최대 simulation 시간 | 명시 없음 | `Unspecified — Saving` | $20\,\mathrm{s}$ | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:34), target 값 아님 |
| failure 후 평가 window | joint lock 이후 생존을 측정하는 최대 시간 | 최대 $20\,\mathrm{s}$ | `Paper Explicit — Saving`, Sec. IV-D–E, pp. 5–7 | failure 후 $20\,\mathrm{s}$ 확보 | 논문 그대로; 별도 선택 없음 |

## 파라미터 표 B — observation, history, action, PD

| 파라미터 | 짧은 설명 | 확인된 사실·값 | 근거 분류·정확한 위치 | 이번 구현 선택 | 선택 근거 |
|---|---|---|---|---|---|
| sensor state 차원 | 현재 센서만으로 만든 policy 입력의 앞부분 | $\mathbf x_t\in\mathbb R^{30}$; joint encoder, IMU, foot sensor 사용 | `Paper Explicit — Saving`, Sec. III-B, p. 4 | 30차원 유지 | 논문 그대로; feature 배치는 다음 행에서 별도 결정 |
| sensor state 구성 | 30개 원소에 어떤 센서값을 넣는지 | target에는 feature별 차원이 없음 | `Unspecified — Saving`; `Inherited — Saving → RMA`, Sec. IV-A에서 같은 A1의 $12+12+2+4$ 확인 | $[\mathbf q_t-\mathbf q_0\;(12),\dot{\mathbf q}_t\;(12),\mathrm{roll/pitch}\;(2),\mathbf c_t^{\mathrm{foot}}\;(4)]$ | `Implementation choice`; RMA 구성을 계승했으며 target 직접값은 아님 |
| policy observation | 매 control step에 policy가 직접 보는 전체 벡터 | $\mathbf o_t=[\mathbf x_t,\mathbf a_{t-1}]\in\mathbb R^{42}$ | `Paper Explicit — Saving`, Sec. III-B, p. 4 | 정확히 42차원 | 논문 그대로; 별도 선택 없음 |
| observation 원소 순서 | tensor 안에서 feature가 놓이는 고정 index 순서 | 명시 없음 | `Unspecified — Saving` | 위 표기 순서를 schema constant로 고정 | `Implementation choice — Agent recommendation`; checkpoint 호환성 확보 |
| history 길이 | student가 과거 dynamics를 추론할 observation frame 수 | $H=50$ | `Paper Explicit — Saving`, Sec. III-B 및 IV-A, p. 4 | $\mathcal H_t\in\mathbb R^{50\times42}$ | 논문 그대로; $50\,\mathrm{Hz}$에서 $1\,\mathrm{s}$는 `Derived` |
| reset history 채움값 | 새 episode의 존재하지 않는 과거 frame 처리 | target에 없음 | `Unspecified — Saving`; Rapid 공식 코드에는 env별 zero-fill이 있음 | reset된 env의 history만 0으로 초기화 | `Implementation choice`; `Code Experiment — Rapid`, [HistoryWrapper](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/wrappers/history_wrapper.py#L9-L40) |
| joint position noise | encoder position 측정 오차 크기 | noisy sensor라고만 쓰고 분포·크기는 없음 | `Unspecified — Saving` | $\epsilon_q\sim U(-0.01,0.01)\,\mathrm{rad}$ | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:166), target 값 아님 |
| joint velocity noise | encoder velocity 측정 오차 크기 | 분포·크기 없음 | `Unspecified — Saving` | $\epsilon_{\dot q}\sim U(-1.5,1.5)\,\mathrm{rad/s}$ | `Implementation choice`; WIM Code Default, target 값 아님 |
| roll/pitch noise | IMU 자세 측정 오차 크기 | 분포·크기 없음 | `Unspecified — Saving` | $\epsilon_{\mathrm{rp}}\sim U(-0.05,0.05)\,\mathrm{rad}$ | `Implementation choice — Agent recommendation`; WIM gravity noise와 단위가 달라 직접 계승하지 않음 |
| foot-contact noise | 발 접촉 sensor의 false positive/negative 처리 | 분포·확률 없음 | `Unspecified — Saving` | first pass에서는 contact flip 없음 | `Implementation choice — Agent recommendation`; noise 변수 하나를 줄임 |
| action 의미 | 12개 network 출력이 제어기에서 무엇을 뜻하는지 | desired joint position $\hat{\mathbf q}_t\in\mathbb R^{12}$ | `Paper Explicit — Saving`, Sec. III-B, p. 4 | desired joint position action 사용 | 논문 그대로; 별도 선택 없음 |
| action scale·offset | normalized action을 실제 joint angle로 바꾸는 크기와 중심 | target에 없음 | `Unspecified — Saving` | $\hat{\mathbf q}_t=\mathbf q_0+0.25\,\mathrm{clip}(\mathbf a_t,-1,1)$ | `Implementation choice`; [WIM A1 Code Default](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:53), target 값 아님 |
| PD torque 식 | desired angle과 measured angle 차이를 torque로 변환 | $\boldsymbol\tau_t=K_p(\hat{\mathbf q}_t-\mathbf q_t)+K_d(\dot{\hat{\mathbf q}}_t-\dot{\mathbf q}_t)$, $\dot{\hat{\mathbf q}}_t=0$ | `Paper Explicit — Saving`, Sec. III-B, p. 4 | 식 그대로 구현 | 논문 그대로; gain 숫자는 다음 행에서 별도 결정 |
| nominal PD gains | position error와 velocity damping의 세기 | target 숫자 없음 | `Unspecified — Saving` | $K_p=20\,\mathrm{N\,m/rad}$, $K_d=0.5\,\mathrm{N\,m\,s/rad}$ | `Implementation choice`; [WIM A1 Code Default](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:53). RMA의 $55/0.8$은 Raisim 값이라 미사용 |

## 파라미터 표 C — joint-lock FailureEnv

| 파라미터 | 짧은 설명 | 확인된 사실·값 | 근거 분류·정확한 위치 | 이번 구현 선택 | 선택 근거 |
|---|---|---|---|---|---|
| episode당 failure 수 | 한 episode에 몇 번 joint lock을 발생시킬지 | single joint-lock failure | `Paper Explicit — Saving`, Sec. III-A, p. 3 | 최대 1회 | 논문 그대로; 별도 선택 없음 |
| failure joint $J_f$ | 잠길 12개 관절 중 하나를 고르는 규칙 | $J_f\sim U\{1,\ldots,12\}$ | `Paper Explicit — Saving`, Sec. III-A, p. 3 | 12개 관절 균등 표본화 | 논문 그대로; 별도 선택 없음 |
| failure time 분포 형태 | episode 중 lock이 발생할 시점의 확률분포 | $T_f\sim U(T_{\min}^{f},T_{\max}^{f})$ | `Paper Explicit — Saving`, Sec. III-A, p. 3 | uniform 형태 유지 | 논문 그대로; bounds는 다음 행에서 별도 결정 |
| failure time bounds | failure가 너무 이르거나 늦게 발생하지 않게 하는 시간 범위 | $T_{\min}^{f}$와 $T_{\max}^{f}$ 숫자 없음 | `Unspecified — Saving` | $T_{\min}^{f}=2\,\mathrm{s}$, $T_{\max}^{f}=8\,\mathrm{s}$ | `Implementation choice — Agent recommendation`; sensitivity 필수 |
| tolerance 분포 식 | locked joint가 움직일 수 있는 반폭의 원래 표본 식 | $\theta_{\mathrm{tol}}\sim\mathcal N(0,\theta_{\max}^2)$ | `Paper Explicit — Saving`, Sec. III-A, p. 3 | 원식과 raw sample을 provenance log에 보존 | 논문 식 그대로 기록; 음수 처리와 scale은 다음 행에서 별도 결정 |
| executable tolerance | 음수가 될 수 없는 실제 joint-limit 반폭 | $\theta_{\max}$ 숫자와 음수 표본 처리 없음 | `Ambiguous/Unspecified — Saving` | $\epsilon\sim\mathcal N(0,0.075^2)$, $\theta_{\mathrm{tol}}=\mathrm{clip}(\lvert\epsilon\rvert,0.005,0.15)$ | `Implementation choice — Agent recommendation`; 공개 코드에서 찾은 값이 아님 |
| failure center $\bar\theta$ | lock interval의 중심이 되는 failure 순간 관절각 | $\bar\theta=q_{J_f}(T_f)$ | `Paper Explicit — Saving`, Sec. III-A, p. 3 | failure 직전 simulator joint position snapshot | 논문 그대로; 별도 선택 없음 |
| allowed interval | locked joint가 움직일 수 있는 lower/upper 범위 | $[\bar\theta-\theta_{\mathrm{tol}},\bar\theta+\theta_{\mathrm{tol}}]$로 limit overwrite | `Paper Explicit — Saving`, Sec. III-A, p. 3 | 논문 interval 적용 | 논문 그대로; URDF 교집합은 다음 행의 별도 안전 선택 |
| URDF-limit 교집합 | 새 lock interval이 robot의 원래 물리 limit을 넘지 않게 하는 처리 | target에 없음 | `Unspecified — Saving` | 새 lower/upper와 original URDF limit의 교집합 사용 | `Implementation choice — Agent recommendation`; invalid bound 방지 |
| failure flag 의미 | teacher에게 정상/고장 관절을 알려주는 상태값 | reset 시 $f_t=0$, failure 후 $f_t=J_f$ | `Paper Explicit — Saving`, Sec. III-A, p. 3 | 의미 그대로 유지 | 논문 그대로; encoding은 다음 행에서 별도 결정 |
| failure flag encoding | scalar flag를 network 입력 범위로 바꾸는 방법 | encoding·normalization 없음 | `Unspecified — Saving` | privileged vector에 $f_t/12$ 제공 | `Implementation choice — Agent recommendation` |
| lock 중 actuator torque | 관절이 잠겨도 motor가 지지 torque를 내는지 | locked actuator에도 torque 유지 | `Paper Explicit — Saving`, Sec. III-A, p. 3 | effort-mode PD torque 계속 적용 | 논문 그대로; [WIM effort-mode Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:99)로 구현 |
| reset 시 limit 복구 | 다음 episode에 이전 lock이 남지 않도록 하는 절차 | $f_t=0$만 명시; 복구 순서 없음 | `Unspecified — Saving` | reset 전에 12개 original lower/upper 복구 | `Implementation choice — Agent recommendation`; failure leakage 방지 |
| training termination | 넘어졌다고 판단해 episode를 끝내는 조건 | 명시 없음 | `Unspecified — Saving` | WIM base-contact termination만 유지 | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:138), target 값 아님 |

## 파라미터 표 D — terrain, domain randomization, reset

| 파라미터 | 짧은 설명 | 확인된 사실·값 | 근거 분류·정확한 위치 | 이번 구현 선택 | 선택 근거 |
|---|---|---|---|---|---|
| terrain 종류 | 학습에 노출할 지면 category | smooth slope, rough slope, discrete obstacle | `Paper Explicit — Saving`, Sec. IV-A, p. 4 | 세 종류만 사용 | 논문 그대로; geometry와 비율은 다음 행에서 별도 결정 |
| terrain geometry·비율 | slope 각도, roughness, obstacle 크기와 sampling 비중 | 숫자·비율 없음 | `Unspecified — Saving`; generator 계보는 `Saving → WIM` | 현재 WIM generator 범위에서 각 종류를 $1/3$로 시작 | `Implementation choice`; geometry는 WIM Code Default, $1/3$은 Agent recommendation |
| terrain height-map 사용 | teacher가 주변 지형 높이를 privileged input으로 보는지 | $\mathbf m_t$ 포함 | `Paper Explicit — Saving`, Sec. III-C, p. 4 | teacher에만 제공, student에는 미노출 | 논문 역할 그대로; sample 크기는 다음 행에서 별도 결정 |
| height-map sample grid | 주변 지형을 몇 점에서 측정할지 | 크기·간격 없음 | `Unspecified — Saving` | $17\times11=187$ samples | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:53), target 값 아님 |
| terrain curriculum | 성공도에 따라 terrain 난이도를 올리는 규칙 | 명시 없음 | `Unspecified — Saving` | WIM distance-based curriculum 유지 | `Implementation choice — WIM Code Default`; no-curriculum 비교 |
| friction DR 사용 | 접촉 마찰계수를 episode마다 바꿀지 | friction을 randomize | `Paper Explicit — Saving`, Sec. III-B, p. 4 | enable | 논문 그대로; 범위는 다음 행에서 별도 결정 |
| friction range | 표본화할 마찰계수 범위 | 숫자 없음 | `Unspecified — Saving` | $\mu\sim U(0.5,1.25)$ | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:121), target 값 아님 |
| payload DR 사용 | base mass에 추가 질량을 바꿀지 | payload를 randomize | `Paper Explicit — Saving`, Sec. III-B, p. 4 | enable | 논문 그대로; 범위는 다음 행에서 별도 결정 |
| payload range | robot base에 더하거나 뺄 질량 범위 | 숫자 없음 | `Unspecified — Saving` | $\Delta m\sim U(-1,1)\,\mathrm{kg}$ | `Implementation choice`; WIM Code Default 범위지만 현재 기본은 disabled |
| motor-strength DR 사용 | actuator torque scale을 관절별로 바꿀지 | motor strength를 randomize | `Paper Explicit — Saving`, Sec. III-B, p. 4 | enable | 논문 그대로; 범위·구현은 다음 행에서 별도 결정 |
| motor-strength range | nominal torque 대비 actuator 세기 배율 | target 숫자·구현 없음 | `Unspecified — Saving`; `Code Experiment — Rapid`의 다른 논문/Go1 config에 $[0.9,1.1]$ | $s_m\sim U(0.9,1.1)$, torque에 관절별 곱 | `Implementation choice — Analogous code`; [Rapid Go1 공식 코드](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/go1/go1_config.py#L88-L101), target 값 아님 |
| PD-gain DR 사용 | controller stiffness와 damping을 episode마다 바꿀지 | $K_p,K_d$를 randomize | `Paper Explicit — Saving`, Sec. III-B, p. 4 | enable | 논문 그대로; 범위는 다음 행에서 별도 결정 |
| PD-gain range | nominal gain 대비 random scale 범위 | 숫자 없음 | `Unspecified — Saving` | $K_p/K_{p,0},K_d/K_{d,0}\sim U(0.8,1.2)$ | `Implementation choice — Agent recommendation`; 공개 근거 없음 |
| reset joint pose | episode 시작 시 관절 자세와 perturbation | 명시 없음 | `Unspecified — Saving` | 현재 A1 기본 pose 주변 WIM reset 유지 | `Implementation choice`; [WIM Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:377), target 값 아님 |
| external push | 학습 중 robot base에 외력을 가할지 | 명시 없음 | `Unspecified — Saving` | first pass에서는 disable | `Implementation choice — Agent recommendation`; target 외 변수를 제거 |

## 파라미터 표 E — teacher, student, PPO

| 파라미터 | 짧은 설명 | 확인된 사실·값 | 근거 분류·정확한 위치 | 이번 구현 선택 | 선택 근거 |
|---|---|---|---|---|---|
| teacher input 구조 | simulation에서만 아는 dynamics·state·terrain 정보의 큰 묶음 | $\mathbf e_t=[\mathbf d_t,\mathbf s_t,\mathbf m_t]$, $\mathbf s_t=[\mathbf v_t,\boldsymbol\omega_t]$ | `Paper Explicit — Saving`, Sec. III-C, p. 4 | 세 묶음 구조 유지 | 논문 그대로; 세부 schema는 다음 행에서 별도 결정 |
| privileged schema | $\mathbf d_t$와 $\mathbf m_t$에 어떤 원소를 어떤 순서로 넣는지 | 세부 항목·순서·정규화·차원 없음 | `Unspecified — Saving` | 뒤 MDP 절의 234차원 schema | `Implementation choice — Agent recommendation`; WIM height grid만 Code Default |
| teacher latent dimension | privileged state를 압축한 environment embedding 크기 | $\mathbf z_t=\mu(\mathbf e_t)\in\mathbb R^8$ | `Paper Explicit — Saving`, Sec. III-C 및 IV-A, p. 4 | $D=8$ | 논문 그대로; 별도 선택 없음 |
| student latent dimension | observation history에서 추정한 embedding 크기 | $\hat{\mathbf z}_t=\phi(\mathcal H_t)\in\mathbb R^8$ | `Paper Explicit — Saving`, Sec. III-C, p. 4 | $50$-frame history $\rightarrow8$차원 | 논문 그대로; 별도 선택 없음 |
| teacher MLP | privileged vector를 latent로 압축하는 network 폭 | hidden $[512,256,128]$, ELU | `Paper Explicit but symbol typo — Saving`, Sec. IV-A, p. 4 | $[512,256,128]\rightarrow8$ | 원문의 “respectively” 문구에 따른 해석; 해석임을 유지 |
| policy MLP | observation과 latent를 action으로 바꾸는 network 폭 | hidden $[256,128]$, ELU | `Paper Explicit but symbol typo — Saving`, Sec. IV-A, p. 4 | $[\mathbf o_t,\mathbf z'_t]\rightarrow[256,128]\rightarrow12$ | $\pi(\mathbf z'_t,\mathbf o_t)$ 식과 “respectively”를 함께 적용 |
| student encoder 유형 | 시간 history를 처리할 network 종류 | 1-D CNN | `Paper Explicit — Saving`, Sec. IV-A, p. 4 | 1-D CNN 사용 | 논문 그대로; layer 숫자는 다음 행에서 별도 결정 |
| student CNN layers | channel, kernel, stride와 frame encoder 폭 | target에 없음 | `Unspecified — Saving`; `Inherited — Saving → RMA`, Sec. IV-B | frame MLP $42\rightarrow32\rightarrow32$; Conv1d $(32,32,k=8,s=4)$, $(32,32,k=5,s=1)$, $(32,32,k=5,s=1)$; linear $\rightarrow8$ | `Implementation choice`; RMA 구조 계승, target 직접값 아님 |
| fused latent 식 | teacher에서 student로 policy 입력을 전환하는 보간식 | $\mathbf z'_t=\alpha\hat{\mathbf z}_t+(1-\alpha)\mathbf z_t$ | `Paper Explicit — Saving`, Sec. III-C, p. 4 | 식 그대로 구현 | 논문 그대로; 별도 선택 없음 |
| $\alpha$ schedule 방향 | teacher 의존도를 줄이고 student-only로 가는 방향 | $\alpha=0$에서 시작해 학습 중간쯤 student-only | `Paper Explicit — Saving`, Sec. III-C–IV-C | 단조롭게 $0\rightarrow1$ | 방향은 논문 그대로; 구간은 다음 행에서 별도 결정 |
| $\alpha$ schedule 구간 | 보간을 시작·종료할 정확한 학습 비율 | 숫자 없음 | `Unspecified — Saving` | 0–10%: $0$, 10–50%: linear $0\rightarrow1$, 이후 $1$ | `Implementation choice — Agent recommendation` |
| adaptation loss 식 | student latent를 teacher latent에 맞추는 auxiliary loss | $L=L_{\mathrm{RL}}+\beta L_{\mathrm{adapt}}$, $L_{\mathrm{adapt}}=\lVert\mathbf z_t-\hat{\mathbf z}_t\rVert_2$ | `Paper Explicit — Saving`, Sec. III-C, p. 4 | 식 그대로 구현 | 논문 그대로; reduction과 gradient 처리는 다음 행에서 별도 결정 |
| adaptation reduction·detach | sample loss 집계법과 teacher 쪽 gradient 차단 여부 | Saving에는 명시 없음. RMA Algorithm 1은 $\theta_\phi$만 갱신하고, Rapid 공식 코드는 target encoder를 `torch.no_grad()`로 계산 | `Unspecified — Saving`; `Inherited — RMA`; `Code Experiment — Rapid official` | batch mean $L_2$, teacher target detach | 사용자 결정 + 계보 근거. teacher는 adaptation loss에서만 차단하며 PPO 경로는 유지 |
| $\beta$ 관계 | adaptation loss의 시간별 가중 방향 | $\beta$는 $\alpha$와 음의 상관 | `Paper Explicit — Saving`, Sec. III-C | 음의 상관 유지 | 방향은 논문 그대로; 함수는 다음 행에서 별도 결정 |
| $\beta$ schedule 값 | 각 학습 시점의 실제 adaptation weight | 함수·최댓값 없음 | `Unspecified — Saving` | transition에서 $\beta=1-\alpha$, 이후 $0$ | `Implementation choice — Agent recommendation` |
| critic input·architecture | PPO value function이 볼 정보와 network 크기 | Saving에는 전혀 명시 없음. Rapid 공식 코드는 $V([\mathbf o_t,\mu(\mathbf e_t)])$와 hidden $[512,256,128]$ 사용 | `Unspecified — Saving`; `Inherited/Code Experiment — Rapid official` | $V([\mathbf o_t,\mu(\mathbf e_t)])$, MLP $[512,256,128]\rightarrow1$ 권장 | target 직접값은 아님. symmetric $V([\mathbf o_t,\mathbf z'_t])$는 ablation |
| RL algorithm | policy optimization 방법 | PPO 사용 | `Paper Explicit — Saving`, Sec. III-C–IV-A | PPO 유지 | 논문 그대로; 숫자는 다음 행에서 별도 결정 |
| PPO hyperparameters | rollout, epoch, clip, entropy, discount, learning-rate 값 | target 숫자 없음 | `Unspecified — Saving` | rollout 24, epochs 5, minibatches 4, clip 0.2, entropy 0.01, $\gamma=0.99$, $\lambda=0.95$, adaptive LR $10^{-3}$, KL 0.01 | `Implementation choice`; [WIM/rsl_rl Code Default](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:202), target 값 아님 |
| 논문 teacher 학습량 | 논문 비교에서 teacher에 제공한 simulation step 수 | $600\,\mathrm{M}$ steps | `Paper Explicit — Saving`, Sec. IV-B, p. 5 | 최종 허용 상한 $600\,\mathrm{M}$ | 논문 수치를 상한으로 사용 |
| 중간 학습 gates | 큰 학습 전에 중단·확장 여부를 판단할 step 수 | 논문에 없음 | `Unspecified — Saving` | $25\,\mathrm{M}$ smoke $\rightarrow150\,\mathrm{M}$ trend $\rightarrow600\,\mathrm{M}$ | `Implementation choice — Agent recommendation` |

## MDP/환경 명세

### MDP 항목별로 따라간 근거 경로

| MDP 항목 | 탐색 경로 | 직접 확인한 것 | 끝까지 찾지 못한 것 | first-pass 처리 |
|---|---|---|---|---|
| deployable observation | `Saving`, Sec. III-B → 참고문헌 [7] `RMA`, Sec. IV-A | `Saving`: 42차원 식; `RMA`: 같은 A1의 30차원 sensor 구성 | target의 feature 순서·정규화 | RMA 구성 계승 + 순서는 Agent recommendation |
| action·PD control | `Saving`, Sec. III-B → 현재 `WIM` A1 config | `Saving`: desired joint position과 PD 식; `WIM`: action scale·gain | target의 action scale, $K_p$, $K_d$ | WIM Code Default 사용 |
| joint-lock transition | `Saving`, Sec. III-A → 저자 프로젝트·Git 이력·Rapid 포크 | failure 식, limit overwrite, torque 유지 | $T_{\min}^{f}$, $T_{\max}^{f}$, $\theta_{\max}$ 및 공식 구현 | Agent recommendation + sensitivity |
| terrain·reset | `Saving`, Sec. IV-A → 참고문헌 [21] `WIM` → 현재 WIM code | target terrain 이름; WIM terrain generator/reset 구조 | target geometry·비율·curriculum | WIM Code Default + 비율은 Agent recommendation |
| domain randomization | `Saving`, Sec. III-B → [21] `WIM` code → [9] `Rapid` official code | randomize 대상; 두 코드의 실제 범위 | target 분포·범위·resampling timing | WIM 보수 범위 우선, Rapid Go1 값은 analogous 후보 |
| reward | `Saving`, Sec. III-B → 참고문헌 [7] `RMA`, Sec. III-A | target의 목적·$v_x^\star$; RMA의 수식·scale | target의 정확한 식·weight | RMA 식 계승, forward cap만 target 값 적용 |
| teacher privileged state | `Saving`, Sec. III-C → [21] `WIM` height grid | $\mathbf e_t=[\mathbf d_t,\mathbf s_t,\mathbf m_t]$와 $\mathbf s_t$ 구성 | $\mathbf d_t$ 세부 항목과 height-map 차원 | WIM grid + 명시적인 Agent schema |
| history student | `Saving`, Sec. III-C/IV-A → [7] `RMA` → [9] `Rapid` official code | $H=50$, 1-D CNN, RMA layer 구조, Rapid reset 처리 | target CNN kernel/channel과 reset fill | RMA CNN 계승 + Rapid zero-fill |
| PPO·critic | `Saving`, Sec. III-C → [9] `Rapid` 공식 코드 → [21] `WIM`/`rsl_rl v1.0.2` | PPO와 joint loss 식; Rapid critic은 observation+teacher privileged latent; 현재 PPO defaults | target PPO 숫자·critic 구조 | WIM defaults + Rapid 계보 asymmetric critic |

### 상태, 관측, 행동을 WIM tensor로 옮기기

`Saving`에서 직접 확인되는 deployable observation은 다음 식뿐이다.

$$
\mathbf o_t=[\mathbf x_t,\mathbf a_{t-1}]\in\mathbb R^{42},
\qquad \mathbf x_t\in\mathbb R^{30}.
$$

`Saving`, Sec. III-B, p. 4에는 $\mathbf x_t$가 joint encoder, IMU, foot sensor에서 온다고 적혀 있지만 feature별 차원과 순서는 없다. `Saving → RMA`, Sec. IV-A를 한 단계 타고 가면 같은 Unitree A1에서 정확히 $12+12+2+4=30$ 구성을 확인할 수 있다. 따라서 first pass는 다음처럼 고정한다.

$$
\mathbf x_t=
\left[
\underbrace{\mathbf q_t-\mathbf q_0}_{12},
\underbrace{\dot{\mathbf q}_t}_{12},
\underbrace{\rho_t,\psi_t}_{\text{roll, pitch}:2},
\underbrace{\mathbf c_t^{\mathrm{foot}}}_{4}
\right],
$$

$$
\mathbf o_t=
\left[
\mathbf q_t-\mathbf q_0,
\dot{\mathbf q}_t,
\rho_t,
\psi_t,
\mathbf c_t^{\mathrm{foot}},
\mathbf a_{t-1}
\right].
$$

이 두 번째 식은 `Saving` 원문 그대로가 아니라 **RMA에서 feature 구성을 계승하고 순서를 제가 정한 recommendation**이다. 현재 WIM의 [`compute_observations()`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:209)은 base velocity, projected gravity, command, 187개 height sample까지 넣어 235차원을 만들기 때문에 그대로 사용할 수 없다. `A1LimpingEnv.compute_observations()`에서 42차원 tensor를 새로 만들고, slice 상수를 한 파일에서 관리해야 한다.

history는

$$
\mathcal H_t=\left[\mathbf o_{t-H+1},\ldots,\mathbf o_t\right]
\in\mathbb R^{50\times42},\qquad H=50
$$

로 정의한다. $H=50$은 `Saving`, Sec. III-B/IV-A의 explicit 값이다. 반면 reset 시 zero-fill은 논문에 없으며, `Rapid` 공식 코드의 [HistoryWrapper](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/wrappers/history_wrapper.py#L9-L40) 구현을 차용한다.

행동과 torque는 `Saving`, Sec. III-B의 식을 그대로 따르되, scale만 현재 WIM A1 코드에서 가져온다.

$$
\hat{\mathbf q}_t=\mathbf q_0+0.25\,\mathrm{clip}(\mathbf a_t,-1,1),
$$

$$
\boldsymbol\tau_t
=K_p(\hat{\mathbf q}_t-\mathbf q_t)
+K_d(\dot{\hat{\mathbf q}}_t-\dot{\mathbf q}_t),
\qquad \dot{\hat{\mathbf q}}_t=\mathbf 0.
$$

$0.25$, $K_p=20$, $K_d=0.5$는 모두 [`A1RoughCfg.control`](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:53)의 `Code Default`이며 `Saving` 값은 아니다.

### Teacher privileged vector

`Saving`, Sec. III-C, p. 4에서 직접 확인되는 식은

$$
\mathbf e_t=[\mathbf d_t,\mathbf s_t,\mathbf m_t],
\qquad
\mathbf s_t=[\mathbf v_t,\boldsymbol\omega_t]
$$

이다. $\mathbf d_t$의 정확한 항목·순서·차원과 $\mathbf m_t$의 크기는 명시되지 않았다. first pass의 recommendation은 다음과 같다.

$$
\begin{aligned}
\mathbf d_t={}&[
\mu_t,\Delta m_t,
\mathbf s_t^{\mathrm{motor}}(12),
\mathbf s_t^{K_p}(12),
\mathbf s_t^{K_d}(12),\\
&f_t/12,\bar\theta_t,\theta_{\mathrm{tol},t}],\\
\mathbf s_t={}&[\mathbf v_t^{\mathrm{body}}(3),\boldsymbol\omega_t^{\mathrm{body}}(3)],\\
\mathbf m_t={}&\text{WIM의 }17\times11\text{ local height samples}.
\end{aligned}
$$

이 schema의 총 차원은 $234$다. 다음을 구분해야 한다.

- $\mathbf d_t$, $\mathbf s_t$, $\mathbf m_t$, failure flag 자체는 `Paper Explicit — Saving`, Sec. III-C/IV-A다.
- 187개 height sample은 [`WIM Code Default`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot_config.py:53)다.
- $\bar\theta_t$와 $\theta_{\mathrm{tol},t}$를 privileged input에 포함하는 것은 **Agent recommendation**이다. `Saving`은 이를 명시하지 않는다.
- 아직 발생하지 않은 $J_f$와 $T_f$를 teacher에게 미리 공개하지 않는 것도 **Agent recommendation**이다. 미래 failure leakage를 피하기 위한 선택이다.

### FailureEnv를 실제 코드로 만드는 과정

`Saving`, Sec. III-A, p. 3의 failure model은

$$
T_f\sim U(T_{\min}^{f},T_{\max}^{f}),
\qquad
J_f\sim U\{1,\ldots,12\},
\qquad
\theta_{\mathrm{tol}}\sim\mathcal N(0,\theta_{\max}^2)
$$

이다. 식은 explicit이지만 $T_{\min}^{f}$, $T_{\max}^{f}$, $\theta_{\max}$는 논문·공개 코드 어디에도 없다. first-pass의 $U(2,8)\,\mathrm{s}$와 half-normal tolerance는 **제가 만든 recommendation**이며 반드시 config와 log에 남겨야 한다.

WIM에 통합하는 순서는 다음과 같다.

1. `A1LimpingEnv._init_buffers()`에서 env별 $T_f$, $J_f$, $\theta_{\mathrm{tol}}$, $f_t$, $\bar\theta_t$ buffer를 만든다.
2. `reset_idx(env_ids)`의 첫 단계에서 이전 failure가 바꾼 lower/upper limit을 원 URDF 값으로 되돌린다. 원 limit은 현재 [`_process_dof_props()`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:279)가 읽으므로 별도 immutable copy를 보존한다.
3. reset 후 $f_t=0$으로 만들고 해당 env만 $T_f$, $J_f$, $\theta_{\mathrm{tol}}$을 다시 표본화한다.
4. [`_post_physics_step_callback()`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:320)에서 처음으로 $t\ge T_f$가 된 env만 찾는다.
5. physics state에서 $\bar\theta=q_{J_f}(T_f)$를 snapshot하고,

$$
q_{J_f}^{\min}=\max(q_{J_f,\mathrm{URDF}}^{\min},\bar\theta-\theta_{\mathrm{tol}}),
\qquad
q_{J_f}^{\max}=\min(q_{J_f,\mathrm{URDF}}^{\max},\bar\theta+\theta_{\mathrm{tol}})
$$

로 actor DOF property를 바꾼다. URDF limit과 교집합을 취하는 부분은 안전을 위해 추가한 **Agent recommendation**이다.
6. $f_t=J_f$로 갱신하되 기존 effort-mode PD torque는 계속 보낸다. 이것이 torque가 사라지는 free-swing failure와의 차이다.
7. `set_actor_dof_properties`는 newly-failed/reset env에만 호출하고, 4,096 env에서 호출 비용을 계측한다.

$\theta_{\mathrm{tol}}$이 음수가 될 수 있는 원문 모순은 그대로 구현할 수 없다. first pass는

$$
\epsilon\sim\mathcal N(0,0.075^2),
\qquad
\theta_{\mathrm{tol}}=\mathrm{clip}(\lvert\epsilon\rvert,0.005,0.15)
$$

를 사용한다. 이는 다른 공개 코드에서 찾은 값이 아니라 **Agent recommendation**이다. raw $\epsilon$과 변환된 $\theta_{\mathrm{tol}}$을 둘 다 기록하고, 이후 $\sigma\in\{0.025,0.075,0.15\}$ sensitivity로 결론 의존성을 확인한다.

### Reset, termination, transition

환경 transition은 기존 [`LeggedRobot.step()`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:80)의 decimation loop를 유지한다. 변경되는 부분은 control step boundary에서 failure event를 발생시키는 callback과 reset 복구뿐이다.

- training termination: `Saving`에는 threshold가 없다. 따라서 현재 WIM의 base collision termination만 유지한다 (`Code Default`).
- reset pose: `Saving`에 없다. 현재 WIM의 A1 pose/reset을 우선 사용하되 source를 `WIM Code Default`로 기록한다.
- training episode: `Saving`에 없다. 현재 WIM의 $20\,\mathrm{s}$를 사용한다.
- evaluation: `Saving`, Sec. IV-D–E는 failure 후 최대 $20\,\mathrm{s}$를 explicit하게 적는다. 따라서 timeout은 $T_f^{\mathrm{actual}}+20\,\mathrm{s}$로 설정한다.
- failure 전에 넘어진 episode는 `pre_failure_fall`로 별도 집계해 post-failure survival에 섞지 않는다. 이는 **Agent recommendation**이다.

### Reward: Saving에 없는 식을 어떻게 채우는가

`Saving`, Sec. III-B, p. 4에 직접 적힌 것은 $v_x^\star=0.5\,\mathrm{m/s}$와 “stable/smooth forward motion을 장려하고 다른 축 운동, 큰 acceleration, power consumption, collision을 벌점”이라는 설명뿐이다. 정확한 식과 weight는 전부 `Unspecified`다.

`Saving → RMA`, Sec. III-A를 타고 가면 아래 10개 식과 scale을 찾을 수 있다. 따라서 first pass는 RMA 식을 **Inherited**로 사용하되, forward cap만 `Saving`의 $0.5\,\mathrm{m/s}$로 바꾼다.

| 항목 | 짧은 설명 | first-pass 식 | 실제 출처 | target과의 차이 |
|---|---|---|---|---|
| forward | 목표 전진 속도까지 빠르게 걷도록 보상 | $r_{\mathrm{fwd}}=20\min(v_{x,t},0.5)$ | `Paper Explicit — RMA`, Sec. III-A; scale 20 | RMA cap $0.35$를 Saving explicit $0.5$로 변경 |
| lateral/yaw | 옆 미끄러짐과 불필요한 yaw 회전을 억제 | $r_{\mathrm{lat}}=-21(v_{y,t}^2+(\omega_{t}^{\mathrm{yaw}})^2)$ | `Paper Explicit — RMA`, Sec. III-A | `Saving` exact 식 아님 |
| work | 큰 actuator work와 에너지 소비를 억제 | $r_{\mathrm{work}}=-0.002\lvert\boldsymbol\tau_t^\top(\mathbf q_t-\mathbf q_{t-1})\rvert$ | `Paper Explicit — RMA`, Sec. III-A | `Saving`의 power penalty를 근사 |
| ground impact | frame 사이 접촉력 급변을 억제 | $r_{\mathrm{impact}}=-0.02\lVert\mathbf f_t-\mathbf f_{t-1}\rVert_2^2$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| torque smoothness | torque가 control step마다 급격히 변하는 것을 억제 | $r_{\Delta\tau}=-0.001\lVert\boldsymbol\tau_t-\boldsymbol\tau_{t-1}\rVert_2^2$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| action magnitude | 과도한 joint target command를 억제 | $r_a=-0.07\lVert\mathbf a_t\rVert_2^2$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| joint speed | 불필요하게 빠른 관절 운동을 억제 | $r_{\dot q}=-0.002\lVert\dot{\mathbf q}_t\rVert_2^2$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| orientation | 몸체의 roll과 pitch 기울어짐을 억제 | $r_{\mathrm{ori}}=-1.5(\rho_t^2+\psi_t^2)$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| vertical term | 몸체의 불필요한 수직 운동을 억제 | $r_z=-2.0(v_{z,t})^2$ | `Paper Explicit — RMA`, Sec. III-A | RMA는 “Z Acceleration”이라 부르지만 식은 $v_z$라 모호; true acceleration variant와 비교 |
| foot slip | 접촉 중인 발이 지면에서 미끄러지는 것을 억제 | $r_{\mathrm{slip}}=-0.8\lVert\mathrm{diag}(\mathbf c_t^{\mathrm{foot}})\mathbf v_t^{\mathrm{foot}}\rVert_2^2$ | `Paper Explicit — RMA`, Sec. III-A | 계승값 |
| collision objective | thigh/calf의 비정상 지면 접촉을 억제 | thigh/calf collision penalty 포함 | `Paper Explicit — Saving`, Sec. III-B; 정확한 식·scale은 없음 | target의 목적만 그대로 적용 |
| collision implementation | collision 목적을 WIM reward 함수로 수치화 | $r_{\mathrm{collision}}=-1.0\,N_{\mathrm{thigh/calf\ contact}}$ | scale $-1.0$은 `WIM Code Default`; contact count 식은 `Agent recommendation` | target의 명시 수치가 아닌 first-pass 구현 |

RMA는 초기에 penalty를 작게 두고 점진적으로 키운다고 명시하지만 정확한 함수는 target에 없다. 초기 학습 30% 동안 multiplier를 $0\rightarrow1$로 선형 증가시키는 것은 **Agent recommendation**이다. 별도로 현재 WIM reward를 control arm으로 학습해, RMA 계승 reward 선택이 결과를 독점하지 않는지 확인한다.

## WIM 파일별 통합 설계

### 한 control step에서 실제로 일어나는 일

1. episode reset 시 WIM이 robot pose, terrain, friction/payload 등을 정하고, FailureEnv는 추가로 $(T_f,J_f,\theta_{\mathrm{tol}})$을 표본화한다.
2. simulator tensor에서 $\mathbf q_t$, $\dot{\mathbf q}_t$, roll/pitch, foot contact를 읽어 30차원 $\mathbf x_t$를 만들고, 이전 action을 붙여 42차원 $\mathbf o_t$를 만든다.
3. $\mathbf o_t$를 history buffer에 밀어 넣는다. teacher 학습 때만 clean privileged vector $\mathbf e_t$도 별도로 만든다.
4. teacher는 $\mathbf z_t=\mu(\mathbf e_t)$, student는 $\hat{\mathbf z}_t=\phi(\mathcal H_t)$를 만들며, 현재 schedule의 $\alpha$로 $\mathbf z'_t$를 합성한다.
5. policy가 $\mathbf a_t=\pi(\mathbf o_t,\mathbf z'_t)$를 출력하고 이를 desired joint position $\hat{\mathbf q}_t$로 변환한다.
6. WIM의 decimation loop가 네 번의 $200\,\mathrm{Hz}$ physics step 동안 PD torque $\boldsymbol\tau_t$를 적용해 policy는 $50\,\mathrm{Hz}$로 동작한다.
7. control-step boundary가 $T_f$를 처음 넘으면 해당 actor의 $J_f$ limit만 현재 관절각 주변으로 좁힌다. 이후에도 PD torque는 계속 계산된다.
8. 새 state에서 reward·termination을 계산하고 rollout에 $\mathbf o_t$, $\mathcal H_t$, $\mathbf e_t$, action, value, log-probability를 저장한다. rollout이 차면 custom PPO가 policy loss와 $\beta L_{\mathrm{adapt}}$를 함께 갱신한다.

즉 기존 WIM의 physics/terrain/parallel-env 기반은 유지하고, 환경 쪽에는 **관측 재정의·history·privileged state·joint-limit event**를, 학습 쪽에는 **두 encoder·latent fusion·adaptation loss**를 추가하는 작업이다.

### 1. 환경 계층: 가장 먼저 구현할 부분

| 새 파일·수정 파일 | 기존 WIM 연결점 | 구체적으로 구현할 내용 | 완료 판단 |
|---|---|---|---|
| `legged_gym/envs/a1_limping/a1_limping_config.py` | [`A1RoughCfg`](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:33) 상속 | `env.num_observations=42`, privileged dimension, 고정 $v_x^\star$, 3종 terrain, failure/DR/reward config를 한곳에 선언 | 모든 recommendation과 source label을 resolved config로 dump |
| `legged_gym/envs/a1_limping/a1_limping.py` | [`LeggedRobot`](/home/jihun/legged_gym/legged_gym/envs/base/legged_robot.py:52) 상속 | failure buffer, limit 변경·복구, 42-D observation, privileged vector, reward term 구현 | 학습 없이 deterministic failure fixture 통과 |
| `legged_gym/envs/a1_limping/history.py` | `Rapid`의 HistoryWrapper 구조 참고 | $\mathcal H_t\in\mathbb R^{50\times42}$ shift와 env별 reset zero-fill | cross-env contamination 0 |
| [`legged_gym/envs/__init__.py`](/home/jihun/legged_gym/legged_gym/envs/__init__.py:47) | 현재 task registry | `a1_limping_base`, `a1_limping_failure`를 같은 class와 다른 config로 등록 | 두 task가 기존 `train.py --task=...`에서 생성됨 |

`BaseEnv`와 `FailureEnv`는 별도 대규모 class 두 개로 만들 필요가 없다. 같은 `A1LimpingEnv`에서 `cfg.failure.enabled`만 다르게 해, failure 이외의 observation/reward/terrain이 완전히 같도록 유지하는 편이 비교에 안전하다.

### 2. 학습 계층: 환경 검증 뒤 구현할 부분

| 새 파일·수정 파일 | 기존 WIM/rsl_rl 한계 | 구체적으로 구현할 내용 | 참고 구현 |
|---|---|---|---|
| `legged_gym/learning/limping_actor_critic.py` | stock `ActorCritic`에는 $\mu$, $\phi$, fused latent가 없음 | teacher encoder, RMA CNN student, policy, privileged critic, student-only inference method | `Saving` Sec. III-C/IV-A + `RMA` Sec. IV-B |
| `legged_gym/learning/limping_storage.py` | stock storage는 observation/privileged observation만 저장 | $\mathcal H_t$, $\mathbf e_t$, rollout 당시 $\alpha$를 저장 | `Rapid` 공식 [rollout storage](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym_learn/ppo/rollout_storage.py) |
| `legged_gym/learning/limping_ppo.py` | stock PPO에는 $L_{\mathrm{adapt}}$가 없음 | PPO loss와 $\beta L_{\mathrm{adapt}}$ 결합, fused latent와 log-prob 일치 보장 | `Saving` 식이 기준; `Rapid`의 [separate adaptation update](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym_learn/ppo/ppo.py#L156-L170)는 구조 참고만 함 |
| `legged_gym/learning/limping_runner.py` | stock runner에 $\alpha/\beta$ schedule과 history 전달이 없음 | iteration 경계에서 schedule 갱신, checkpoint에 schedule/schema 저장 | 수치 schedule은 Agent recommendation |
| [`legged_gym/utils/task_registry.py`](/home/jihun/legged_gym/legged_gym/utils/task_registry.py:104) | line 147에서 `OnPolicyRunner`를 직접 생성 | config의 runner name에 따라 stock 또는 limping runner를 선택 | 외부 `rsl_rl` package는 직접 patch하지 않음 |
| `legged_gym/scripts/evaluate_limping.py` | 기존 `play.py`는 failure matrix 평가 기능 없음 | 고정 $T_f,J_f,\theta_{\mathrm{tol}}$ 입력, raw JSON/CSV 저장 | 환경/학습 완성 뒤 구현 |

stock [`rsl_rl v1.0.2`](https://github.com/leggedrobotics/rsl_rl/tree/2ad79cf0caa85b91721abfe358105f869a784121)는 asymmetric critic은 지원하지만 history, teacher/student fused latent, adaptation loss는 지원하지 않는다. 따라서 config만 바꾸는 것으로는 `Saving`을 구현할 수 없다. 반면 `Rapid` 공식 코드는 history와 adaptation module을 구현하지만 separate adaptation update이며 `Saving`의 joint $\alpha/\beta$ training과 동일하지 않다.

## 논문에 수치가 없을 때의 처리 규칙

파라미터를 임의로 채운 뒤 논문값처럼 사용하는 것을 막기 위해 아래 우선순위를 코드 config와 결과 manifest에도 그대로 남긴다.

| 상황 | 문서·config 표기 | 선택 방법 | 이후 조치 |
|---|---|---|---|
| `Saving`에 식과 숫자가 모두 있음 | `Paper Explicit — Saving`, section/page | 그대로 구현 | parser/unit test로 수식과 단위 확인 |
| `Saving`에는 개념·식만 있고 숫자가 없음 | `Unspecified — Saving` | `Saving`이 직접 인용한 논문에서 같은 로봇·같은 변수의 값을 우선 검색 | 찾으면 `Inherited — Saving → 논문명`으로 기록하고 target 값이 아님을 명시 |
| 인용 논문보다 공식 코드에만 값이 있음 | `Code Experiment` 또는 `Code Default` | paper-specific 실행 config인지 단순 base default인지 구분 | 저장소 URL, commit, 파일·행, 대응 논문을 함께 기록 |
| 유사 논문/다른 로봇 코드에만 값이 있음 | `Analogous code — 논문/로봇명` | 단위와 물리적 범위를 검토한 뒤 시작점으로만 사용 | target 설정처럼 말하지 않고 반드시 sensitivity/ablation 수행 |
| 어떤 근거에도 값이 없음 | `Agent recommendation` | 현재 WIM에서 안전하게 실행되는 보수적 값 선택 | resolved config에 표시하고 범위 실험으로 결론 의존성 확인 |
| 구현 중에도 결정할 근거가 없음 | `Blocked parameter` | 해당 기능을 끄거나, 결론을 내리지 않는 최소 구현 사용 | 이 문서의 사용자 질문에 올리고 답을 받기 전 target reproduction 주장 금지 |

예를 들어 $T_f\sim U(T_{\min}^{f},T_{\max}^{f})$는 `Saving`에 식만 있고 양 끝값은 없다. 그래서 $T_{\min}^{f}=2\,\mathrm{s}$, $T_{\max}^{f}=8\,\mathrm{s}$는 **Agent recommendation**으로 시작하고, 값 자체를 config key `source: agent_recommendation`과 함께 저장한다. 반대로 $s_m\sim U(0.9,1.1)$은 `Rapid`의 공식 Go1 코드에서 가져온 **다른 논문·다른 로봇의 Analogous code**다. 두 경우 모두 `Paper Explicit — Saving`으로 승격해서는 안 된다.

## 구현 단계와 gate

### 0. 재현 manifest와 deterministic foundation

- 모든 run에 git SHA, Isaac Gym version, rsl_rl SHA, seed, resolved config, observation schema hash를 저장한다.
- per-env `torch.Generator` 또는 seed-derived sampling을 사용해 env ordering이 바뀌어도 $(T_f,J_f,\theta_{\mathrm{tol}})$ fixture가 재현되게 한다.
- 42-D observation/history/privileged dimension test를 먼저 통과시킨다.

Gate: 2회 실행의 first 100-step observation/failure trace가 bitwise 또는 허용 오차 내 동일.

### 1. BaseEnv privileged teacher를 먼저 학습하기

- failure를 끄고 $\alpha=0$, $L_{\mathrm{adapt}}=0$으로 둔다. student encoder와 history는 학습 경로에서 제외한다.
- teacher는 simulator에서 얻는 clean state, DR parameter, terrain height map 등 현재 정의된 privileged ground truth $\mathbf e_t$를 모두 받고, $\mu(\mathbf e_t)\in\mathbb R^8$로 압축해 policy에 전달한다. policy에 raw ground truth를 직접 concat하는 별도 oracle 구조는 만들지 않는다.
- $v_x^\star=0.5\,\mathrm{m/s}$ 고정 명령과 42차원 deployable observation으로 flat terrain부터 64 env smoke train을 수행한다.
- current reward와 RMA-derived reward를 각각 짧게 학습해 넘어짐/zero-action collapse/NaN을 비교한다.
- flat gate 후 세 terrain과 conservative DR을 하나씩 켜서 teacher encoder가 privileged variation을 실제로 이용하는지 확인한다.

Gate: failure 없는 evaluation에서 $20\,\mathrm{s}$ survival $\ge90\%$, warm-up 이후 median $v_x\ge0.35\,\mathrm{m/s}$, PPO/critic/teacher gradient와 latent가 finite. 미달이면 FailureEnv teacher나 joint training으로 넘어가지 않는다.

### 2. joint-lock mechanics를 학습 없이 검증하기

- 학습기 변경 없이 결정론적인 $(J_f,T_f,\theta_{\mathrm{tol}})$ fixture를 주입한다.
- selected joint position이 allowed bound를 넘지 않는지, non-selected joint property가 불변인지, torque가 계속 발생하는지 확인한다.
- reset 후 12개 original limits가 모두 복원되는지 확인한다.
- 4,096 env에서 dynamic property API latency와 missed-event count를 측정한다.

Gate: 12 joints×3 center poses×3 tolerances fixture 100% 통과; reset leakage 0; failure event exactly once/episode. 이 단계는 환경 correctness 검사이며 policy 성능 판정이 아니다.

### 3. FailureEnv privileged teacher를 학습하기

- BaseEnv teacher와 같은 $\alpha=0$ teacher-only 구조, observation, reward, terrain, conservative DR, PPO configuration을 사용하고 `failure.enabled`만 켠다.
- teacher privileged state에는 현재 failure flag, failed joint identity, lock center/width와 clean simulator state를 포함한다. 아직 발생하지 않은 future $J_f,T_f$는 공개하지 않는다.
- 논문의 teacher 비교처럼 최종 비교 run은 각 환경에서 scratch로 학습한다. BaseEnv checkpoint warm-start는 mechanics/debug smoke에만 허용하고 결과 표에는 사용하지 않는다.
- RMA-derived reward 각 항의 scale, raw value, weighted value를 따로 log해 failure penalty 하나가 전체 reward를 지배하는지 확인한다.
- failure 발생 전/후의 관측, reward, joint limit, torque trace를 저장해 환경 mechanics를 사람도 읽을 수 있게 만든다.

Gate: 동일 failure matrix에서 FailureEnv teacher가 BaseEnv teacher보다 survival/고장 후 velocity 경향이 높고, no-failure 성능 감소를 함께 보고한다. FailureEnv teacher 자체가 걷지 못하면 student/joint training으로 넘어가지 않는다.

### 4. teacher 단계 동결과 재현 run

- BaseEnv/FailureEnv 모두 64-env smoke 후 $25\,\mathrm{M}\rightarrow150\,\mathrm{M}$ gate를 통과시킨다. 필요할 때만 최대 $600\,\mathrm{M}$으로 확장한다.
- 각 teacher는 scratch, 동일 seed set, 동일 PPO/environment configuration으로 학습한다. Saving 논문도 teacher들을 scratch에서 같은 환경·PPO 설정으로 학습했다고 명시한다.
- FailureEnv teacher에서 failure flag 포함/제거 비교를 수행한다.
- 통과한 observation/reward/terrain/DR/failure schema와 teacher trainer를 `teacher-v1`로 동결한다.

Gate: 두 teacher의 최소 3 seed 결과와 checkpoint가 보존되고 FailureEnv teacher 우위 경향이 확인됨. 그 전에는 joint-training 결과를 본 실험으로 해석하지 않는다.

### 5. history student separate-transfer sanity check

- [7] CNN으로 teacher latent를 detached target으로 맞춘다.
- teacher rollout과 student on-policy rollout 양쪽에서 latent MSE, $L_2$와 action disagreement를 측정한다.
- 이것은 논문의 `[SS]` 비교군이자 joint trainer 디버깅 oracle이다.

Gate: held-out seed에서 latent $L_2$가 random student 대비 $50\%$ 이상 감소하고 student inference가 NaN 없이 $20\,\mathrm{s}$ 실행.

### 6. joint teacher-student training

- 먼저 BaseEnv에서 짧은 joint-training smoke를 수행해 failure dynamics 없이 $\alpha/\beta$, detach, fused latent와 PPO ratio를 검증한다.
- 그다음 BaseEnv[S][JT]와 FailureEnv[S][JT]를 동일 총 step/seed/config로 각각 scratch에서 학습한다. FailureEnv[S][JT]가 주 실험이고 BaseEnv[S][JT]가 비교군이다.
- rollout iteration 동안 $\alpha$와 $\beta$는 고정하고 iteration 경계에서만 갱신한다.
- PPO action log-probability는 rollout 때 사용한 fused latent와 같은 $\alpha$로 재계산한다.
- total loss, PPO surrogate/value/entropy, $L_{\mathrm{adapt}}$, teacher/student latent norm, gradient norm을 각각 log한다.
- checkpoint에는 schedule progress와 history schema를 저장한다. 최종 export는 $\phi+\pi$만 포함한다.

Gate: midpoint부터 $\alpha=1$이며, 이후 teacher input을 제거한 evaluation 결과가 동일 checkpoint의 teacher-forced 결과와 함께 저장됨. stock PPO 대비 custom loss unit test 통과.

### 7. 환경 완성 후 평가와 ablation

공통 failure matrix를 미리 생성하고 모든 정책에 재사용한다.

- policy: BaseEnv teacher/student, FailureEnv teacher/student, failure flag 제거, separate supervision, joint training
- terrain: smooth slope, rough slope, discrete obstacle 각각 1,500 instance
- stratification: 12 joints, hip/thigh/calf, failure time bin, tolerance bin, terrain level
- metrics: failure 전후 $v_x$, $\Delta v_x$, $20\,\mathrm{s}$ survival mean/$P_{25}$/$P_{50}$, $5/10/20\,\mathrm{s}$ survival rate, fall reason, torque saturation fraction, locked-joint bound violation, power/work, recovery time
- uncertainty: seed별 원자료, bootstrap 95% CI, paired difference CI
- ablation: $\sigma\in\{0.025,0.075,0.15\}$, failure time bounds, DR off/on, terrain curriculum off/on, failure flag, lock center/width privileged inclusion, asymmetric/symmetric critic, $\alpha$ schedule, $\beta_{\max}$, current vs RMA-derived reward. no-detach는 논문 재현 arm이 아니라 gradient-path 진단용 선택 실험으로만 둔다.

논문 Table I의 0.57→0.47 m/s 및 survival 56.5/P25 10.8/P50 59.0은 참고선으로만 overlay한다. simulator/version/미지정 parameter가 다르므로 일치 여부를 성공 조건으로 삼지 않는다.

## 성공 기준

필수 성공 기준은 다음과 같다.

1. mechanics correctness: locked joint bound violation P99 ≤0.01 rad, reset limit leakage 0, single-event invariant 100%.
2. baseline viability: no-failure $20\,\mathrm{s}$ survival $\ge90\%$, median forward speed $\ge0.35\,\mathrm{m/s}$.
3. emergence evidence: 공통 seed/failure matrix에서 joint-trained FailureEnv student가 BaseEnv student보다 primary failure metrics에서 더 높고, 논문의 FailureEnv 우위 경향과 같은 방향을 보인다. 고정된 $+10$ percentage-point 문턱이나 논문 수치 일치는 요구하지 않는다. seed별 차이와 paired bootstrap $95\%$ CI는 효과의 불확실성을 숨기지 않기 위해 함께 보고한다.
4. utility preservation: FailureEnv student의 pre-failure speed 감소가 BaseEnv student 대비 $20\%$ 이내.
5. student deployability: privileged input을 제거한 student-only JIT path가 $50\,\mathrm{Hz}$ deadline을 만족하고 teacher tensor access test가 실패하도록 구성.
6. reproducibility: 최소 3 training seeds, resolved config/checkpoint/evaluation matrix/raw metric 보존.

숫자가 논문과 다르다는 이유만으로 구현 실패로 판정하지 않는다. 다만 평균 하나만 우연히 앞서는 결과를 피하기 위해 terrain/joint/seed별 결과를 함께 공개하며, 방향이 seed에 따라 뒤집히거나 CI가 매우 넓으면 “경향 미확정/추가 seed 필요”로 구분한다.

## 결정론적 테스트 목록

- observation slice 값/단위/noise mask, exact dimension 42
- history shift, reset zero-fill, cross-env contamination 없음
- failure sampler uniform joint histogram과 fixed-seed exact sample
- negative normal 처리/clip/URDF-bound intersection
- failure 전 limits 불변, failure 순간 center snapshot, 이후 position bound
- nonfailed joints/actors 불변
- reset/timeout/pre-failure fall 모두 limits 복구
- privileged vector에는 clean data, student observation에는 privileged data 없음
- $\alpha\in\{0,0.5,1\}$에서 fused latent exact equality
- $\beta$ schedule monotonic negative correlation
- adaptation loss 단독 backward에서 teacher encoder gradient는 0/None이고 student encoder gradient는 nonzero인지 확인; PPO backward에서는 $\alpha<1$일 때 teacher encoder gradient가 존재하는지 별도 확인
- PPO ratio/log-probability가 rollout $\alpha$와 일치
- student-only export가 environment factor argument 없이 실행
- survival percentile 계산을 hand-crafted episode fixture와 비교

## 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| target 공식 코드/수치 부재 | exact reproduction 주장 불가 | 모든 선택을 config/provenance에 기록, sensitivity 중심 결론 |
| runtime DOF property API 병목 | 대규모 FPS 급락 | newly-failed/reset env만 호출, latency 계측; 필요 시 failure time bucket은 별도 deviation으로만 사용 |
| $\theta_{\mathrm{tol}}$ 음수 모순 | invalid lower/upper | half-normal transform을 명시하고 raw sample 저장, 대안 ablation |
| teacher privileged leakage | 가짜 student 성능 | student-only signature/test/JIT export, evaluation에서 privileged buffer 제거 |
| joint loss instability | latent collapse/NaN | teacher target detach, adaptation/PPO gradient 경로 분리 test, gradient/norm log, staged SS sanity check |
| reward 불명 | walking collapse 또는 비논문 gait | current reward control arm + RMA-derived arm, penalty curriculum, term-by-term log |
| current A1 URDF/PD와 paper asset 차이 | torque/lock behavior 차이 | URDF SHA와 joint order 저장, physical conclusion 금지 |
| discrete obstacle가 A1에 과도 | aggregate 결과 왜곡 | terrain별 결과를 먼저 보고 aggregate는 동일 가중치로만 계산 |
| 600M 비용 | iteration 지연 | 25M/150M gates, failing configuration 조기 중단 |

## 아직 Searcher가 해결하지 못한 연구 항목

- $T_{\min}^{f}$, $T_{\max}^{f}$, $\theta_{\max}$
- target reward 식/weight와 penalty curriculum
- exact observation order/noise/normalization
- privileged vector와 height map dimension
- terrain geometry/proportions/curriculum
- DR distribution과 resampling timing
- $\alpha/\beta$ schedule과 $\beta$ scale
- Saving target의 critic architecture/input, PPO hyperparameter, final student total steps/seeds
- critical failure definition과 physical trial count

공개 paper/project/author repository/인용 공식 코드를 모두 확인한 범위에서는 더 이상 target-specific 수치가 나오지 않았다. 저자 config/checkpoint가 제공되면 이 목록부터 교체한다.

## 사용자 결정 상태와 남은 질문

1. **재현 성공의 우선순위 — 결정됨**: simulation-only first pass. 실제 A1 soft/hard lock은 범위에서 제외한다.
2. **실패 tolerance 해석 — 위임받아 결정함**: half-normal + bounds clip을 사용하고 raw sample과 변환값을 기록한다.
3. **미지정 failure time — 위임받아 결정함**: $T_f\sim U(2,8)\,\mathrm{s}$로 시작하며 $T_f\in\{0,2,5,8\}\,\mathrm{s}$ 고정 평가와 bounds ablation을 둔다.
4. **관측 30-D 구성 — 결정됨**: 논문 차원/센서 문구와 맞는 [7]의 $[\mathbf q_t,\dot{\mathbf q}_t,\mathrm{roll/pitch},\mathbf c_t^{\mathrm{foot}}]$ 구성을 사용한다.
5. **reward 계보 — 결정됨**: Saving의 명시 목적과 $0.5\,\mathrm{m/s}$를 우선하고, 빠진 식/scale은 RMA/Rapid 계보에서 provenance와 함께 가져온다.
6. **DR 폭 — 결정됨**: conservative current ranges로 먼저 수렴시키고 wide DR은 후속 ablation으로 확장한다.
7. **teacher target gradient — 결정됨**: adaptation loss에서는 $\mathbf z_t$를 detach한다. RMA Algorithm 1은 $\theta_\phi$만 갱신하고 Rapid 공식 코드도 encoder target을 `no_grad`로 계산한다. Saving은 detach를 쓰지 않았다고 명시하지 않으므로 이 계보를 따른다. teacher encoder의 PPO gradient는 $\alpha<1$ 동안 유지한다.
8. **compute budget — 결정됨**: RTX 4090으로 $25\,\mathrm{M}/150\,\mathrm{M}$ gate를 먼저 수행하고 부족할 때 H100으로 확장한다. 최종 상한은 논문의 $600\,\mathrm{M}$ step을 reference로 둔다.
9. **성공 threshold — 결정됨**: exact numerical reproduction은 요구하지 않는다. joint training이 작동하고 FailureEnv가 BaseEnv보다 높은 failure 성능을 보이는 방향성 재현을 필수 기준으로 사용한다. CI와 seed별 결과는 판정의 신뢰도 정보로 보고하되 고정 $+10$ threshold는 두지 않는다.
10. **외부 `rsl_rl` 수정 정책 — 위임받아 결정함**: repository-local custom runner/PPO/model/storage를 추가하고 외부 `rsl_rl`은 수정하지 않는다. 이렇게 해야 environment와 알고리즘 변경 SHA가 한 저장소에 같이 남고 WIM 기본 task도 보존된다.
11. **critic 입력 — 설명 후 확인 가능**: Saving 논문은 critic을 전혀 설명하지 않는다. 선택지는 (a) asymmetric $V([\mathbf o_t,\mu(\mathbf e_t)])$: 학습 중에만 privileged 정보로 value estimation을 쉽게 하고, actor/student inference에는 누출하지 않는 방식, (b) symmetric $V([\mathbf o_t,\mathbf z'_t])$: actor와 같은 정보만 보아 가정은 적지만 critic 난도가 높고 schedule에 따라 입력 분포가 바뀌는 방식이다. [9] Rapid 공식 구현은 (a)를 사용한다. (a)에서 teacher encoder를 critic과 공유하면 value loss도 $\mu$를 갱신하므로, detached target은 iteration 간에는 움직일 수 있다. 이는 stop-gradient 위반이 아니라 joint teacher PPO의 결과다. **권장 기본:** (a)로 시작하고 (b)는 ablation으로 둔다. 이는 Saving의 명시값이 아니라 가장 가까운 공식 계보에서 가져온 구현 선택이다.
