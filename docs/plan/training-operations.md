# Base teacher 학습 운영 기록

기준일: 2026-08-07, RTX 4090 24 GB, `legged_ws`.

## Managed training loop

장기 학습은 `.codex/agents`의 `supervisor`(대화에서는 **슈퍼바이저**)만 시작하거나
resume한다. `tracker`는 3분마다 로컬 지표를 수집하고, 추세 해석 간격은
3→6→12→최대 15분으로 조절한다. hard failure 또는 충분한 burn-in 이후 여러
지표가 함께 정체된 경우에만 정확한 process group을 중단한다. 종료 상태는
슈퍼바이저가 exactly-once claim한 뒤 `ANALYZING` 상태로 `analysis`에 넘긴다. 승인된
계획만 `MODIFYING→VERIFYING`으로 이동하며, 두 검증을 통과한 경우에만 `READY`가
된다. 검증 실패나 agent-stage 오류는 `ERROR`로 되돌아가 incident analysis를
다시 요청한다. 정상 완료도 다음 phase 판단을 위해 analysis 대상이다.

3분 수집은 모델 polling이 아니라 `watch --event-driven` Python process가 담당한다.
이 process는 `latest_snapshot.json`을 계속 갱신하되 hard alert, trend candidate,
지정 checkpoint, terminal 상태에서만 compact event를 출력한다. 한 run에는 tracker
하나만 존재하며, tracker 소유 중 supervisor는 `collect`, `status`, `watch`, checkpoint
polling을 중복 실행하지 않는다. tracker는 `operator` stop을 사용할 수 없고, trend
stop은 manifest의 burn-in(기본 iteration 250)과 보존 checkpoint를 CLI가 강제한다.

모든 managed run의 `launch`, `collect`, `watch`, `stop`은 처음부터 끝까지 동일한
default execution domain에서 실행한다. managed process lifetime 동안
`require_escalated`를 사용하거나 domain을 전환하는 것은 금지한다. supervisor는 child
시작 직후 `supervisor_heartbeat.json`을 atomic write하고 2초마다 갱신한다. collector는
검증된 `/proc` identity 또는 10초 이내 heartbeat만 supervisor 생존 근거로 인정한다.
PID가 보이지 않고 heartbeat도 stale/missing이면 첫 관측에는 RUNNING을 유지하면서
`liveness_probe.json`과 `supervisor_pid_unobservable` warning을 기록한다. 15초 이상 뒤의
두 번째 연속 관측에서도 exit status와 생존 근거가 없을 때만 `missing_supervisor` ERROR로
전환한다. metrics/console 진전은 빠른 재확인을 뒷받침할 뿐 supervisor 생존을 무기한
증명하지 않는다.

`supervisor_pid_unobservable` warning은 같은 default domain에서 즉시 재확인한다. 다른
domain에서 blind stop/retry하지 않는다. stop 호출자가 기록된 PID, PGID, boot ID,
start ticks, command, run directory를 모두 검증할 수 없으면 STOPPING 전환이나 signal 없이
fail closed한다. heartbeat/liveness 구현 변경 뒤에는 반드시 새로운 from-scratch P0를
통과하고 terminal experiment analysis가 READY로 routing되어야 P1을 시작할 수 있다.

### P0 → P1 완료 계약

modifier의 두 단계 검증 성공은 현재 하네스로 새 P0를 한 번 재실행할 자격만 만든다.
P0 통과 선언은 아니다. 슈퍼바이저가 `authority=supervisor`로 처음부터 실행한 새 P0가
종료된 뒤 아래 명령이 성공하고, 그 terminal experiment analysis가 `ready`로 routing된
경우에만 P1을 시작할 수 있다.

```bash
python3 -m legged_gym.harness.cli validate-p0 --run-dir <fresh-p0-run-dir>
```

`validate-p0`의 정확한 계약은 다음과 같다.

- manifest: phase `p00-harness`, `mock=false`, `authority=supervisor`, launch GPU
  preflight에 RTX 4090 기록
- process/console: exit code 0, signal 없음, 비어 있지 않은 `console.log`, GPU PhysX와
  enabled GPU pipeline 기록, hard-error signature 없음
- resolved config: task `a1_limping_base_v2`, 64 env, actor observation 48,
  teacher latent 8, actor input 56, env당 rollout 24 step, 5 learning epochs,
  8 minibatches, 2 iterations, save interval 1, seed 1
- metrics: 정확히 iteration `[0, 1]`, transitions `[1536, 3072]`; 두 record의 모든
  숫자가 finite; 5 epochs × 8 minibatches에 따른 planned/completed PPO update가 각각
  40으로 동일; `nonfinite_update_skipped=0`; `PPO/mean_kl`과 `PPO/max_kl`이 존재하고
  finite이며 0 이상. 설정되지 않은 hard KL 상한은 적용하지 않는다.
- checkpoints: 비어 있지 않은 `model_0.pt`, `model_1.pt`, `model_2.pt`와 남은
  temporary checkpoint 없음

명령은 failures와 warnings를 포함한 JSON report를 출력하며 failure가 하나라도 있으면
nonzero로 종료한다. 종료 후 GPU telemetry 수집 실패는 launch preflight가 성공했다면
warning이며 단독 gate failure가 아니다. `authority=leader`인 과거 run은 변경하지 않고
증거로 보존하되 P0 계약을 만족할 수 없다.

modifier 검증은 슈퍼바이저가 정확히 한 번의 새 from-scratch P0를 실행하는 것만
허용한다. 이전에 validation이 실패한 run은 새 validator로 재검증하거나 READY로
소급 인정할 수 없다. 새 P0 종료 후 별도의 terminal experiment analysis가 READY를
권고해야만 슈퍼바이저가 P1 여부를 결정할 수 있다. P0는 gait, convergence, failure
recovery 또는 production policy quality를 평가하지 않는다.

상태와 산출물은 `logs/managed/<phase>/<experiment>/<run-id>/`에 저장한다.
`manifest.json`, `state.json`, `events.jsonl`, `metrics.jsonl`, checkpoint,
console 및 분석/수정 결과가 한 run 아래에 모이며 `logs/`, `runs/`, `wandb/`는
Git에서 제외한다. 별도의 run별 Markdown은 만들지 않는다.

슈퍼바이저가 실행하는 예시는 다음과 같다. `--` 뒤는 shell 문자열이 아니라 실행할
argument vector다. 하네스는 정확한 `--log_dir`를 자동으로 추가한다.

`a1_official_wim_teacher243_failure`도 동일한 `train.py` managed entrypoint로만
실행한다. 이 task는 actor 입력 243D를 유지하고, episode마다 정상 또는 단일 관절의
ADAPT 열화율 `d ∈ {0.2, 0.4, 0.6, 0.8}`을 적용한다. 기존 BaseEnv checkpoint는
읽기 전용 warm-start이며 덮어쓰지 않는다.

```bash
python3 -m legged_gym.harness.cli launch \
  --authority supervisor \
  --phase p01-plane-teacher \
  --experiment base-v2 \
  --label seed1 \
  -- \
  /home/jihun/Capstone2/miniconda3/bin/conda run --no-capture-output \
  -n legged_ws python -u legged_gym/scripts/train.py \
  --task=a1_limping_base_v2 --headless --sim_device=cuda:0 \
  --max_iterations=3000 --seed=1
```

읽기 전용 상태 확인은 `collect --run-dir <run-dir>` 또는 event-driven `watch`를 사용한다.
수동 중단은 supervisor agent 권한을 대체하지 않으며, 정확히 기록된 process supervisor에만
SIGINT→SIGTERM→최후 수단 SIGKILL 순으로 적용된다.

```bash
python3 -m legged_gym.harness.cli collect --run-dir <run-dir>
python3 -m legged_gym.harness.cli watch \
  --run-dir <run-dir> --event-driven --poll-seconds 180 \
  --checkpoint-interval 100
python3 -m legged_gym.harness.cli stop \
  --authority supervisor --run-dir <run-dir> \
  --kind operator --reason "사용자 요청"
```

현재 graceful iteration-boundary checkpoint 보장은 `a1_limping*` teacher
runner에 한정한다. GPU preflight는 실제 RTX 4090을 요구한다. 슈퍼바이저는 사용자가
알려 준 Morai 동시 실행 여부와 실제 GPU 상태를 함께 보고 env 수를 정하며,
VRAM 점유율 자체가 아니라 안정성, throughput, 비교 가능한 transition 수를
최적화한다.

### BLOCKED 분석 재개

`BLOCKED`에는 일반 상태 전이가 없다. 같은 phase의 슈퍼바이저가 명시적인 사용자
scope 결정을 받았을 때만 현재 state version과 현재 `analysis_plan.json`의 lowercase
SHA-256을 CAS 조건으로 제공해 분석을 정확히 한 번 재개할 수 있다.

```bash
python3 -m legged_gym.harness.cli resume-blocked-analysis \
  --run-dir <blocked-run-dir> --authority supervisor \
  --expected-state-version <version> \
  --expected-plan-sha256 <lowercase-sha256> \
  --scope-decision "사용자가 승인한 새 scope"
```

성공 시 기존 plan의 정확한 bytes를 먼저
`analysis-history/<sha256>.json`에 read-only, digest-addressed archive로 보존하고,
기존 `blocked_reason`, state record, plan path/digest와 scope 결정을 state audit에
기록한 뒤 `BLOCKED vN → ANALYZING vN+1` event 하나를 추가한다. archive가 이미 있으면
같은 bytes인지 확인하며 절대 덮어쓰지 않는다. 이후 analysis agent가 새 plan으로
교체하므로 archive-first 순서는 중단 시에도 기존 근거를 복구 가능하게 한다.
authority, state, version, digest 또는 scope 검증 실패와 replay는 state/event를
변경하지 않고 실패한다. 이 명령은 학습이나 평가를 시작하지 않는다.

현재 `a1_limping_base`와 이를 상속하는 WIM control의 기본 throughput 설정은
`32,768 env / 24 rollout steps / 8 minibatches`다. CLI override 없이 이 값이
적용된다. 4090 benchmark에서 가장 높은 후반 FPS를 낸 설정이다.

`a1_limping_base_v2`는 논문 학습량 비교를 위해 별도로
`8,192 env / 24 rollout steps / 8 minibatches / 3,000 iterations`를 기본값으로
사용한다. 총 `589,824,000` transitions다.

## 로그

`train.py`는 다음을 자동 저장한다.

- TensorBoard event: `logs/<experiment>/<timestamp>_<run_name>/events.out.tfevents.*`
- resolved configuration: 같은 디렉터리의 `resolved_config.json`
- checkpoint: 같은 디렉터리의 `model_<iteration>.pt`

터미널 출력을 실시간으로 보려면 `conda run --no-capture-output`을 사용한다.
TensorBoard는 별도 터미널에서 실행한다.

```bash
/home/jihun/Capstone2/miniconda3/bin/conda run --no-capture-output \
  -n legged_ws tensorboard \
  --logdir /home/jihun/legged_gym/logs \
  --host 127.0.0.1 --port 6006
```

브라우저 주소는 `http://localhost:6006`이다. 원격 호스트라면 SSH local
forwarding이 별도로 필요하다.

## Batch override

CLI에서 다음 값을 override할 수 있게 추가했다.

- `--num_envs`: 병렬 Isaac Gym 환경 수
- `--num_steps_per_env`: PPO iteration당 환경별 rollout step
- `--num_mini_batches`: epoch당 PPO minibatch 수
- `--save_interval`: checkpoint iteration 간격

계산식은 다음과 같다.

```text
rollout batch = num_envs * num_steps_per_env
minibatch size = rollout batch / num_mini_batches
total transitions = rollout batch * max_iterations
```

V2 100-iteration smoke:

```bash
/home/jihun/Capstone2/miniconda3/bin/conda run --no-capture-output \
  -n legged_ws python legged_gym/scripts/train.py \
  --task=a1_limping_base_v2 --headless --sim_device=cuda:0 \
  --max_iterations=100 --seed=1 \
  --run_name=base_v2_smoke100_seed1 \
  --wandb --wandb_project=saving-the-limping \
  --wandb_entity=josephx03021-ajou-univ
```

CLI에서 `--num_envs`를 주지 않으면 v2 기본값 8,192가 적용된다. 100-iteration
최종 smoke는 NaN 없이 끝났고 고정-command 평가에서 20초 survival 100%, 평균
전진 속도 0.171 m/s, path efficiency 0.981을 보였다. 다만 four-feet contact
71.6%, vertical velocity RMS 0.450 m/s라 gait 성공은 아니다.

본 학습은 동일 명령에서 `--max_iterations=3000`을 빼거나 그대로 3,000으로
지정한다. checkpoint 250/500/1,000에서 deterministic 평가를 먼저 하고,
stomping 지표가 개선되지 않으면 terrain/failure 단계로 넘어가지 않는다.

## 4090 capacity benchmark

각 설정을 plane BaseEnv에서 10~20 iteration 실행했다. GPU utilization은
Isaac Gym과 PPO가 실제로 올라간 sample(`VRAM > 4 GB`)만 평균했다. VRAM에는
display/base process가 포함된다.

| env | rollout step | minibatch 수 | rollout batch | minibatch size | active GPU util | peak VRAM | 후반 FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 24 | 4 | 98,304 | 24,576 | 65.5% | 5.3 GB | 195~209k |
| 8,192 | 24 | 8 | 196,608 | 24,576 | 75.5% | 6.1 GB | 243~260k |
| 12,288 | 24 | 8 | 294,912 | 36,864 | 77.8% | 7.2 GB | 276~294k |
| 16,384 | 24 | 8 | 393,216 | 49,152 | 79.5% | 8.3 GB | 290~306k |
| 16,384 | 24 | 16 | 393,216 | 24,576 | 76.4% | 7.5 GB | 290~295k |
| 24,576 | 24 | 8 | 589,824 | 73,728 | 82.1% | 10.4 GB | 304~315k |
| 32,768 | 24 | 8 | 786,432 | 98,304 | 78.2% | 12.5 GB | 327~340k |
| 49,152 | 24 | 8 | 1,179,648 | 147,456 | 74.5% | 16.6 GB | 298~315k |
| 57,344 | 24 | 8 | 1,376,256 | 172,032 | 73.9% | 18.5 GB | 308~316k |
| 57,344 | 24 | 4 | 1,376,256 | 344,064 | 66.6% | 23.4 GB | 305~319k |

GPU utilization 수치만 보면 `24,576/24/8`이 높지만, raw throughput은
`32,768/24/8`이 가장 높아 기본값으로 채택했다. 그러나 큰 minibatch는
baseline PPO optimization을 바꾸므로 최종
재현 run과 capacity/engineering run을 구분해야 한다. 16,384 env와 16
minibatch는 baseline minibatch size를 유지하는 보수적 후보다.

VRAM을 20 GB 가까이 채우기 위한 49k~57k env 설정은 성공적으로 실행됐지만
GPU utilization이 74%대로 내려가고 FPS도 32,768 env보다 낮았다. 따라서 VRAM
occupancy를 목표로 env를 늘리는 것은 이 workload의 학습 시간을 줄이지 않는다.
57,344 env에서 minibatch 수를 4로 줄이면 VRAM은 23.4 GB까지 차지만 GPU
utilization은 66.6%로 더 낮아지고 OOM 여유도 1.1 GB뿐이다. 이는 사용 금지
capacity edge이며 권장 학습 설정이 아니다.

환경 수를 바꾸면 같은 `max_iterations`의 총 transition도 바뀐다.

| 설정 | 25M | 150M | 600M |
|---|---:|---:|---:|
| 4,096 env, 24 step | 255 iter | 1,526 iter | 6,104 iter |
| 16,384 env, 24 step | 64 iter | 382 iter | 1,526 iter |
| 24,576 env, 24 step | 43 iter | 255 iter | 1,018 iter |
| 32,768 env, 24 step | 32 iter | 191 iter | 763 iter |

현재 RMA-derived task의 penalty curriculum은 raw iteration 기반이므로 env 수를
바꾸고 iteration만 줄이면 동일 transition에서 같은 curriculum이 아니다. WIM
reward task에는 이 penalty curriculum이 없다. 논문 비교용 run은 명시된 4,096
env를 유지하거나 curriculum을 transition 기반으로 고친 뒤 수행한다.

## 2026-08-09 bounded-action 및 W&B 교정

기존 run은 unbounded Gaussian sample을 PPO storage/log-prob에 저장하면서,
environment에서는 그 값을 `[-1,1]`로 clip했다. 따라서 PPO가 확률을 계산한
action과 실제 로봇이 실행한 action이 달랐고, raw output이 커진 뒤에는 서로 다른
policy가 같은 clipped action을 실행했다. 새 구현은 `a=tanh(u)`인 squashed
Gaussian과 Jacobian-corrected log-prob을 사용한다. exploration std는 RMA의 공개
하한인 `0.2` 이상으로 유지한다. 이 변경 전 checkpoint는 비교/evaluation에만
사용하고 새 장기 run을 resume하지 않는다.

actor observation의 previous-action도 수정했다. `env.step(a_t)`가 반환하는 다음
observation에 이제 즉시 `a_t`가 들어가며, reset된 episode는 0으로 시작한다.

W&B는 `sync_tensorboard=True`를 제거했다. 32k run에서 scalar별 TensorBoard
tailing이 iteration 733에서 밀린 반면 실제 학습은 3000까지 완료됐기 때문이다.
runner가 iteration마다 모든 scalar를 하나의 `wandb.log` 호출로 직접 전송한다.
추가 진단은 다음과 같다.

- `Policy/raw_mean_abs`
- `Policy/sampled_action_abs`
- `Policy/deterministic_action_abs`
- `Policy/action_near_bound_rate` (`abs(action)>=0.98`)
- `Rollout/mean_forward_velocity`
- `Rollout/reset_rate_per_step`

1,024 env, 20 iteration 회귀 run에서 value loss는 `6.80 -> 0.61`, action
near-bound rate는 `2.20% -> 2.61%`, 평균 전진속도는 `-0.157 -> 0.083 m/s`였고
NaN/Inf는 없었다. 이 짧은 run은 장기 locomotion 성공 판정이 아니라 구현 안정성
검사다.

### 8k run NaN과 PPO 안정화

run `Aug09_03-21-35_base_teacher_bounded_v2_8k_20260809`는 iteration 57
직후 종료됐다. surrogate loss가 iteration 55/56/57에서
`0.122 -> 3.917 -> 4088.364`로 폭증했고 다음 PPO update에서 actor mean이
NaN이 됐다. 같은 구간의 action near-bound rate는 `29.7% -> 31.9% -> 33.4%`로
증가했다. `model_50.pt`의 모든 tensor는 finite하므로 simulator/OOM 문제가 아니라
policy-ratio update의 수치 폭발로 판정했다.

근거 구분은 다음과 같다.

- **Paper Explicit (RMA supplement S1):** 15,000 iterations, batch 80,000,
  4 minibatches, 4 rounds, ratio `[0.8,1.2]`, value coefficient `0.5`, fixed
  learning rate `5e-4`, Gaussian std 하한 `0.2`.
- **Unspecified (Saving the Limping):** PPO learning rate, schedule, desired KL,
  epoch/minibatch와 action distribution의 bounded 처리.
- **Paper Explicit (WIM Appendix A.4):** desired KL `0.01`에 따른 adaptive
  learning rate.
- **Implementation choice:** raw pre-tanh sample을 rollout storage에 보관하고
  Isaac Gym에는 `tanh(raw)`만 전달; KL `0.05`에서 남은 epoch 중단;
  log-ratio를 exponentiation 전에 `[-20,20]`으로 제한; loss/gradient finite
  guard; adaptive learning rate 상한을 초기 `5e-4`, 하한을 `1e-5`로 제한.

PPO ratio에서는 동일한 stored raw sample에 대한 tanh Jacobian이 old/new
log-prob 차이에서 상쇄된다. 따라서 `atanh(bounded_action)` 복원이 필요 없고,
경계 근처 정밀도 손실도 없다.

회귀 run `Aug09_03-42-03_regression_raw_action_kl_8k`은 8,192 env에서 120
iterations(23.59M transitions)를 완료했다. 이전 실패 지점 58을 통과했고 결과는
다음과 같다.

- surrogate loss 범위: `[-0.0094, 0.0101]`
- value loss: `6.378 -> 0.196`
- nonfinite update skip: 0회
- KL hard early-stop: iteration 4, 5의 2회
- mean forward velocity: `-0.153 -> 0.407 m/s`
- 1,500-env deterministic 평가: 평균/중앙 vx `0.457/0.457 m/s`, survival 0%

즉 PPO NaN 회귀는 통과했고 전진 locomotion도 학습했다. 다만 120 iterations는
RMA의 1.2B transitions 대비 2% 수준이며 survival gate는 아직 통과하지 못했다.
장기 run에서는 `PPO/nonfinite_update_skipped=0`, `PPO/max_kl<0.05`, surrogate
loss가 0 근처인지 먼저 확인하고, checkpoint별 20초 survival을 별도로 평가한다.

### Privileged-observation timestep 정렬 교정

24,000-iteration run을 사후 분석한 결과, `TeacherPPO.act()`가 environment-owned
`privileged_obs_buf`의 참조를 transition에 보관하고 `env.step()` 이후에 storage로
복사했다. privileged buffer는 step 중 in-place 갱신되므로 action/old mean은
`e_t`에서 계산됐지만 storage에는 `e_{t+1}`이 기록됐다. policy가 GT에 민감해진
iteration 60부터 parameter update 전 첫 minibatch KL이 약 `0.051`이 되어 모든
update가 중단됐다.

rollout observation과 privileged observation을 action 계산 시점에 `clone()`하도록
수정하고 alias-mutation 단위 테스트를 추가했다. 동일한 `8,192 env / seed 1`의
run `Aug09_18-57-34_regression_obs_alignment_8k_seed1`에서 100 iterations를
검증한 결과:

- `completed_updates=0`인 iteration: 0개
- 계획된 32 update를 모두 완료한 iteration: 66개
- 총 optimizer update: `2,410 / 3,200`
- nonfinite update: 0개
- surrogate loss 범위: `[-0.0094, 0.0184]`
- episode length: `17.6 -> 530.5` step
- rollout forward velocity: `-0.162 -> 0.537 m/s`
- 1,500-env deterministic 20초 평가: survival `100%`, 평균/중앙 vx
  `0.554/0.554 m/s`, action near-bound `13.7%`

따라서 observation alignment gate와 BaseEnv teacher plane gate를 통과했다. 이
결과에서는 RMA-derived MDP를 WIM reward로 교체할 근거가 없으며, 다음 단계는
동일 코드로 더 긴 checkpoint 추세와 terrain gate를 확인하는 것이다.
