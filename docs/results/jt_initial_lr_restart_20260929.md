# JT 시작 학습률 통제 및 TF43000 재출발

2026-09-29. 사용자가 진행 중인 JT를 중지하고 낮은 시작 학습률로 새로 시작할 것을 요청했다. 중단 모델을 재개하지 않는다.

## 기존 실행 중지

- 대상 PID602002, W&B `6o0wx906`.
- 명령행을 확인하고 SIGINT를 전달했다. 기존 signal handler가 iteration 경계에서 저장·종료했다.
- manifest `completed_new_iterations=12039`, stdout `COMPLETE 12039`, 최종 checkpoint `model_55039.pt`.
- 기존 데이터는 지우지 않았다. 초기 LR1e-3 조건의 중간 대조 자료로 보존한다. 완결된 35000-iteration 결과로 취급하지 않는다.

## 변경 하나

`train_jt_student_width_fresh.py`에 명시적 `--initial-learning-rate` 옵션을 추가했다. 옵션이 없으면 기존 동작이다. 이번 실행에서는1e-5를 설정했다. runner 생성 전 config에 적용하고, TF 로드 후 algorithm LR과 모든 optimizer group LR이 설정값과 같은지 검사한다. 시작값을 manifest에 남긴다.

TF43000 가중치, 학생CNN64, seed1, 4096환경, rollout24, 신규35000 iteration, 보상·환경·고장 분포·α/β·adaptive LR 규칙은 유지한다. 새 학생과 새 optimizer로 시작한다. 티처 인코더만 별도 감속하거나 KL guard를 추가하지 않았다. **LR을 학습 내내1e-5에 고정하는 변경은 아니다.**

## 실제 1-iteration 사전 검증

출력: `logs/ablation/jt_width64_lr1e5_preflight_20260929/`.

기존 CNN64 manifest와 비교해 초기 teacher/student/actor 가중치 SHA 및 environment 설정이 일치했다. 학습 설정 차이는 초기 LR, 시험 길이1 iteration, run_name이다. 초기 물리 상태의 전체 해시는 기록되지 않았으므로 정확한 시뮬레이션 상태 일치까지 주장하지 않는다.

| 첫 iteration 지표 | 기존 시작 LR1e-3 | 수정 시작 LR1e-5 |
|---|---:|---:|
| 평균 KL | 122.429274 | 0.006524 |
| 최대 minibatch KL | 144.248413 | 0.009282 |
| PPO ratio clip fraction | 0.949976 | 0.078660 |
| 수행 optimizer update | 20 | 20 |
| 종료 시 LR | 1e-5 | 3.375e-5 |

수정 모델의 첫 저장 checkpoint를 같은 고장 후 입력으로 CPU 추론해 TF43000과 비교했다.

| 입력 | 기존 첫 JT 목표 관절각 MAE | 수정 첫 JT 목표 관절각 MAE | 수정 평균 KL |
|---|---:|---:|---:|
| TF 방문 상태, n1896 | 0.401445 rad | 0.003084 rad | 0.009560 |
| 학생 방문 상태, n1904 | 0.374919 rad | 0.002865 rad | 0.008451 |

동일 입력에서 기존 약21–23°의 출력 변화가 약0.16–0.18°로 감소했다. 실제 관절 추종오차가 아니라 정책 출력 차이다. 계산 기록은 사전 검증 폴더의 `startup_policy_verification.json`. CUDA를 초기화하지 않은 CPU 계산이다.

**시작 LR을 낮추면 첫 정책 급변이 크게 줄어드는 효과는 직접 확인했다.** 43500까지 나타나는 모든 학습 변화가 초기 LR 탓이라는 결론이나 최종 학생 Q 향상 결론은 아니다.

## 본 학습

- PID:1232785. 정확한 command와 stdout 경로는 `logs_jt_width64_lr1e5_fresh_20260929.launch.json`.
- 출력: `logs/jt_wim243_width64_lr1e5_fresh_20260929/run_seed1_35000/`.
- stdout: `logs_jt_width64_lr1e5_fresh_20260929.log`.
- W&B: [dpyvk2sq](https://wandb.ai/josephx03021-ajou-univ/saving-the-limping/runs/dpyvk2sq).
- 본 실행 첫 평균 KL은0.006585로 사전 시험과 비슷했다. 본 실행 manifest도 기존과 비교하여 초기 teacher/student/actor SHA, 환경, seed, 환경 수, 전체 학습 예산이 같고 초기 LR만 변경됐음을 확인했다. 결과는 본 실행 폴더의 `launch_verification.json`.
- TF 원본 SHA256: `944a697abb30dfc8023e15544d0909acfcdaa4d8c4c0f930656847398f150635`.
- 사전 검증 checkpoint를 이어 사용하지 않는다. TF43000부터 새 학생·optimizer로 시작한다.
- 같은 예산인73000과 추가 학습78000을 구분한다. 최종 효과는 학생 단독 Q, d별 Q, 생존 및 속도 RMSE로 평가한다. 원본71500 및73000을 모두 비교 대상으로 둔다.

이 파일의 PID와 진행 상태는 작성 시점의 기록이며, 지속 모니터링 완료를 뜻하지 않는다.
