# JT 체크포인트 진단 후 재학습 계획 v1

2026-09-28. 이 문서는 재학습 제안이며, 아직 학습 설정을 변경하거나 신규 학습을 시작하지 않았다. 근거: `docs/results/jt-transition-audit-20260928.md` 및 `logs/evaluations/jt-transition-audit-20260928/`.

## 1. 목표를 구분한다

1. 학생 encoder의 latent가 고정된 티처 encoder 출력에 가까워지는 것.
2. 학생이 만든 latent와 actor를 조합했을 때 실제 추종/생존이 좋아지는 것.

첫 번째가 좋아져도 두 번째가 자동으로 좋아지는 것은 아니다. 최종 선택 기준은 student-only 생존율 S, 추종 시간 비율 Q, vx/vy/yaw-rate RMSE다. 회복률은 제외한다. 현재 JT encoder는 PPO의 actor·critic 손실과 모방 손실을 함께 받는다. 순수한 지도 모방기와 다르다.

티처는 고장 출력 계수 등 GT를 즉시 보지만 학생은 결과로 나타난 history로 추정한다. 고장 직후 동일 정보가 없으므로 모든 순간의 동일 성능은 보장할 수 없다. 단, 이것으로 고장 후 20초 전체의 성능 격차가 불가피하다고 설명해서도 안 된다.

## 2. 먼저 추가할 관측/평가 항목

- 같은 validation 조건의 student-only S/Q/RMSE를 정기적으로 기록. 혼합 policy reward를 학생 성능으로 부르지 않는다.
- Teacher와 Student latent RMS를 분리해서 기록. 현재 `Latent/mean_abs`는 Teacher 출력이다.
- latent 오차는 raw L2와 상대 RMS를 함께 기록. scale 변화와 입력 상태분포 변화를 구분한다.
- 같은 입력에서의 행동 차이를 기록. 정규화 action 차이 외에 `0.25 × action 차이`인 목표 관절각 차이(rad)도 기록한다.
- 고정된 TF43000 참고 정책의 행동도 따로 저장한다. 현재 JT teacher branch가 원래 TF 정책과 같다고 가정하지 않는다.
- d×joint별 episode 배정 수, 실제 고장 후 transition 수, 종료율, Q를 기록한다. 기존 aggregate 로그만으로 강한 고장의 실제 유효 데이터 부족을 확정할 수 없다.
- 모델 선택용 validation seed와 최종 평가 seed를 분리한다. 마지막 checkpoint 또는 혼합 reward 최댓값만으로 선정하지 않는다.

## 3. 가장 먼저 비교할 변경: Teacher encoder 고정

**가설:** 전환 중 바뀌고 커지는 teacher latent가 학생 모방의 목표를 불안정하게 만든다. TF43000 encoder를 고정하면 같은 물리정보에 대한 목표 표현을 유지할 수 있다.

실험 A:
- 동일 TF43000에서 시작한다.
- Teacher encoder만 고정한다. Student encoder와 shared actor/critic은 계속 학습한다.
- 우선 alpha 10,000-iteration 전환과 beta=1-alpha는 유지한다.
- 보상·고장 분포·네트워크 폭·history 길이·총 iteration은 그대로 둔다.
- fixed reference encoder의 출력 오차와 student-only Q를 함께 평가한다.

이것은 Teacher encoder와 actor를 모두 고정한 분리학습과 다르다. 또한 encoder를 고정해도 shared actor는 바뀌므로 teacher branch 행동 전체가 보존되는 것은 아니다. 원래 TF 행동이 필요하면 frozen reference actor도 별도로 보관한다.

관찰된 목표 변화가 실제 성능 격차의 원인인지는 이 통제 실험 전까지 확정하지 않는다. 결과가 좋아지지 않으면 목표 이동이 주된 병목이라는 가설을 재검토한다.

## 4. 전환 시간 연장은 별도 비교

실험 B:
- 기존 JT 구조와 손실을 유지하고 전환 길이만 10,000→20,000 iteration으로 변경한다.
- 두 run 모두 신규 총 30,000 iteration으로 비교한다. 전환 연장은 학생 단독 학습 시간을 줄이므로 이를 명시한다.
- student-only가 되는 시점과 같은 총 budget 시점 양쪽에서 비교한다.

전환을 늘리면 alpha 증가뿐 아니라 beta 감소도 느려진다. 따라서 현 코드에서는 '제어 전환 속도'와 '모방 가중치 유지 시간'이 함께 바뀐다. 둘을 분리하려면 alpha와 beta의 schedule 설정을 별개로 둬야 한다. 변경 효과를 하나로 과장하지 않는다.

## 5. beta floor 및 행동 모방의 위치

실험 C (A/B의 결과 후): alpha와 독립적으로 beta를 작은 양수로 남기는 비교. 기존 beta-floor=0.1 profile은 구현돼 있으나 효과가 입증된 것은 아니다. GT 목표가 계속 바뀌는 상태에서 beta만 남기는 것은 학생에게 어려운 목표를 계속 강제할 수 있다.

실험 D (필요할 때): 고정 TF43000의 행동을 학생이 실제 방문한 상태에서 질의하고 작은 action-distillation 항을 사용한다.

\[
L_{act}=\mathbb E\left[\|\pi_S(o_t,h_t)-\operatorname{stopgrad}(\pi_{TF}(o_t,p_t))\|_2^2\right].
\]

- latent 8개를 동일하게 맞추는 것보다 행동에 중요한 차이를 직접 줄이는 대안이다.
- 학생이 방문한 상태를 포함해야 한다. 티처의 정상적인 궤적만으로 지도학습하면 학생의 오차 누적 상태를 충분히 다루지 못할 수 있다.
- 잘못된 상태에서도 TF 행동이 항상 최적이라는 보장은 없으므로 큰 고정 가중치를 바로 적용하지 않는다.
- 순차 행동에서의 데이터 분포 이동 문제는 DAgger가 다룬 선행 문제다. 이 프로젝트에서 해당 원인이 실증됐다는 뜻은 아니다: https://proceedings.mlr.press/v15/ross11a.html

## 6. Frozen Teacher/Actor baseline 주의

기존 `FrozenTeacherStudentActorCritic`은 Teacher encoder, actor, action std를 고정하고 Student와 critic을 학습한다. 그러나 현재 critic 입력에 Student latent가 연결되어 있으므로 value loss의 gradient가 Student encoder에 들어갈 수 있다. 따라서 이를 그대로 '순수 supervised latent 모방' 실험이라고 부르면 안 된다.

순수한 encoder 학습 시간의 충분성을 검사하려면 별도 진단에서 다음을 명시해야 한다.
- 고정 Teacher encoder/actor.
- Student에 들어가는 손실을 supervised latent/action loss로 제한하거나, PPO/value gradient 유입을 명확히 분리.
- 학생 유도 상태를 포함한 학습 데이터와 독립 validation 데이터.
- 학습/validation 오차와 폐루프 성능을 함께 확인. 훈련 오차만 작다고 충분히 배웠다고 판정하지 않음.

## 7. 추가 구현 점검

### CNN의 마지막 2프레임

현재 Student CNN은 50프레임(오래된 순→최신 순)을 넣고 첫 Conv1d가 kernel=8, stride=4다. 첫 convolution의 마지막 window는 frame40..47까지이며, 뒤 convolution은 이 출력을 조합한다. 따라서 frame48,49는 latent에 기여하지 않는다. 자동미분 검사에서도 이 두 입력 frame의 gradient가 0임을 확인했다.

- 50Hz 제어에서 encoder history가 최신 40ms를 쓰지 않는다.
- 액터 current235에는 최신 관측이 있으므로 로봇 전체가 40ms 지연되어 있다는 뜻은 아니다.
- 긴 post-fault 구간의 추종 격차를 이 사실만으로 설명할 수는 없다.
- 다음 구현에서는 최근 48프레임을 명시적으로 사용하거나 convolution 정렬을 수정해 최신 frame을 포함시키는 선택을 검토한다. 기존 모델의 입력 의미를 바꾸므로 과거 체크포인트에 몰래 적용하지 않는다.
- 수정은 별도 schema/version과 impulse/gradient 검사로 검증하고, 알고리즘 ablation에는 동일한 history 구현을 사용한다.

### 그 밖의 변경

- onset 2–10초 무작위는 유지한다. 이미 구현되어 있어 수정할 이유가 없다.
- 지형/보상/고장 종류 확대를 이번 원인 분석과 동시에 하지 않는다.
- d=.8/1 oversampling은 실제 post-fault 노출량을 확인한 뒤 비교한다. episode 선택 확률 1/6과 transition 비율을 혼동하지 않는다.
- 현재 학습 에피소드 20초에서 onset이 2–10초이므로 최대 길이로 생존해도 고장 후 경험은 10–18초다. 이번 평가의 고장 후 20초는 더 길다. longer-horizon 학습은 별도 변경 후보지만 우선순위를 encoder 목표/전환 분석보다 앞세우지 않는다.
- 고장 미학습 정상 정책 baseline과 최종 독립 평가가 논문 주장에 필요하다.

## 8. 학부 논문용 실행 순서

1. 평가·로그 기준을 고정한다.
2. 현재 baseline을 보존하고 Teacher-encoder 고정 실험 A를 우선한다.
3. 전환 20,000 실험 B를 별도로 수행해 '시간을 더 주는 효과'를 비교한다.
4. 좋아진 경우만 beta/action loss 등의 추가 실험으로 확장한다. 모든 변경을 한 run에 넣지 않는다.
5. 동일 평가에서 d별 S/Q/RMSE를 보고, 최종 후보 동결 후 독립 seed/지형으로 평가한다.

현재 진단만으로 학생이 티처 성능에 도달할 것을 보장할 수는 없다. 결과가 나쁘면 latent/행동 지도 신호, 실제 노출량, 관측 가능한 정보, 모델 용량을 순서대로 분리해서 조사한다.

## 9. 추종 reward와 Q의 목적 차이

저장된 학습 config의 tracking_sigma는 0.25이며, 선속도 추종 보상은 다음과 같다.

\[
r_v=\exp\left[-\frac{e_x^2+e_y^2}{0.25}\right].
\]

횡오차가 0일 때 전진 오차 .11 m/s는 Q의 .1 기준을 벗어나지만 추종 보상은 약 .953이다. 따라서 Q의 성능 저하를 encoder 모방 문제로만 귀속하면 안 된다. 보상은 연속 오차 및 에너지/움직임 비용의 타협을 학습하며, 세 오차 기준 동시 충족 비율을 직접 최대화한 것이 아니다.

정밀 추종을 연구 목표로 강화한다면 보상 폭·비중의 별도 비교를 고려할 수 있다. 그러나 이번 encoder/schedule 실험과 동시에 바꾸지 않는다. Q를 좋게 보이도록 threshold를 사후 조정하는 것과 학습 목표를 명시적으로 바꾸는 것도 구분한다. 보상 변경 후에는 기존 티처의 목표 행동을 그대로 최적 정답으로 간주할 수 없다는 점도 함께 검토한다.
