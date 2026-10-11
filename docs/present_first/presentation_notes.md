# v12 변경과 발표 질문 답변

입력은 사용자가 첨부한 `/home/jihun/Downloads/legged_first_presentation_v11.pptx`이다. 42쪽의 사용자 수정본을 보존하고, encoder 출처 부록을 추가한 43쪽 v12를 생성했다. 사용자가 추가한 물리적18쪽을 유지하고 제목/설명을 정책 입력과 관절 명령에 맞췄다. 중복된 제목 번호와 하단 번호를 실제 순서에 맞췄다.

## 변경 위치

- 18쪽: Actor 입력243D와 action scale0.25 의미 보완. 기존 공식 수식 유지.
- 22쪽: Fixed / onset 의미와 현재 고장 발생 시점 설명.
- 27쪽: 논문 참고치와 JT 생존 지표의 숫자 비교 보완. 조건 차이 명시.
- 39쪽: Saving the Limping 결과. 목표 속도 유지와 평균 생존 시간 중심.
- 40쪽: RMA 결과. Robust/RMA/Expert 성공률·평균 생존·거리 중심.
- 41쪽: ADAPT 결과. 완전 출력 상실의 두 calf 조건에서 reward 비교 중심.
- 43쪽: Appendix A10, encoder 구조의 논문 근거와 프로젝트 변경.
- 모든 슬라이드·발표자 노트에서 논문명은 `Saving the Limping`으로 통일.

## 1. Encoder의 구체적인 출처

출처 확인 경로는 `Saving the Limping §IV-A → 인용 [7] RMA §IV-B → 현재 joint_teacher_student_actor_critic.py`다. 프레임 입력의 구성은 `joint_schema.py → WIM 관측 구현`을 확인했다.

| 구성 | 논문 근거 | 현재 프로젝트 | 판정 |
|---|---|---|---|
| Teacher 은닉층 | Saving the Limping §IV-A: [512,256,128], ELU | 45→512→256→128→8 | 은닉층/활성화는 Paper Explicit; 입력45는 Implementation choice |
| Teacher latent | Saving the Limping: D=8 | 8D | Paper Explicit |
| Student history | Saving the Limping: H=50, 1D CNN, RMA 인용 | 50×48 | 길이50은 Paper Explicit; 프레임48은 Implementation choice |
| 시간 CNN | RMA §IV-B: (in,out,kernel,stride) = (32,32,8,4), (32,32,5,1), (32,32,5,1) | 동일 채널·커널·stride의 Conv1d3층 | Inherited: Saving the Limping → RMA |
| 프레임 임베딩 | RMA는 32D를 만드는 2층 MLP로 설명 | Linear(48,32)+ELU, 선형층1개 | Implementation choice; 원문과 동일 구조라고 하지 않음 |
| 최종 투영 | RMA: CNN을 펼친 후 선형 투영 | 32×3=96→Linear→8 | 선형 투영은 Paper Explicit;96은 현재 구조에서 Derived |

RMA 자신의 환경 encoder는 현재 Teacher 은닉층과 다르게 기술되어 있다. 따라서 현재 Teacher의 [512,256,128]을 RMA에서 그대로 가져왔다고 설명하면 부정확하다. 현재 프로젝트의 Actor 은닉층도 별도 설정이므로, encoder와 Actor의 출처를 섞지 않는다.

Encoder 가중치는 학습한다. 개발자는 구조·입력·8D 출력과 손실을 정하고, 보행 보상과 모방 손실의 gradient로 가중치를 최적화한다. 8개 성분을 특정 물리량 8개로 지정하거나, 입력을 다시 복원하는 autoencoder를 사용한 것은 아니다. JT 시작 시 Teacher는 TF43000에서 초기화하고 Student는 새로 초기화한다.

발표 답변:

> Teacher의 은닉층과 latent8은 Saving the Limping의 구성을 참고했습니다. Student의 시간 CNN은 그 논문이 계승한 RMA의 커널과 stride를 사용했습니다. 다만45차원 물리정보,48차원 프레임,프레임 임베딩은 현재 WIM 기반 프로젝트에 맞게 구성했으며 완전한 원문 복제라고 주장하지 않습니다.

## 2. 0.25는 각도 단위 변환 계수가 아니다

도에서 라디안으로 바꾸는 식은 다음이다.

\[
q_{\mathrm{rad}}=q_{\mathrm{deg}}\frac{\pi}{180},\qquad\frac{\pi}{180}\approx0.0174533.
\]

현재 식은 다른 역할이다.

\[
q^{\mathrm{des}}_{j,t}=q_j^{\mathrm{default}}+(0.25\,\mathrm{rad})a_{j,t}.
\]

`a`는 무차원 네트워크 출력이며, 0.25는 action을 각도 보정량으로 바꾸는 고정 action scale이다. 기존 `A1RoughCfg.control.action_scale=0.25`를 계승했으며 학습으로 얻거나 단위 변환에서 유도한 값이 아니다. 별도 탐색으로 최적임을 증명한 것도 아니다.

예: 앞 thigh의 기본각0.8rad, action0.4이면 목표각0.9rad다. 0.25×0.4=0.1rad를 기본각에 더한다. 기본각은 고정된 기준 자세이고 직전 목표각을 뜻하지 않는다. 최대 보정각은 action 범위/clipping까지 봐야 하므로 0.25만으로 제한 범위를 말하지 않는다.

## 3. 논문 결과를 직관적으로 읽는 방법

### Saving the Limping

목표 속도0.5m/s에서 고장 후 평균 속도는 일반 BaseEnv0.42 → 고장 대응 FailureEnv0.47m/s다. 목표의84% →94%를 유지한 것이다. 평균 생존 시간은8.94 →11.30초로2.36초 길어졌다. 이 값들은 관절 잠금 조건의 원문 Table I All terrain 결과다.

**주의할 해석:** 평균 속도0.47은 RMSE0.03이라는 뜻이 아니다. 정규화 평균 생존56.5%는 전체의56.5%가20초를 완주했다는 뜻도 아니다.

### RMA

RMA 성공률73.5%는 일반 강건 정책 Robust62.4%보다11.1%p 높고, 실제 환경 정보를 쓰는 Expert76.2%보다2.7%p 낮다. 정규화 평균 생존은 Robust80% / RMA85% / Expert86%, 이동거리는1.13 /1.34 /1.35m다. 즉 RMA는 해당 시뮬레이션 시험에서 일반 강건 정책을 개선했고, 정보가 더 많은 기준 정책에 가까운 결과를 냈다.

### ADAPT

같은 논문 안의 reward로 보면 완전 출력 상실 LFC에서 Single Policy24.22, ADAPT26.44, Teacher26.54다. RFC에서는25.82,26.42,26.50이다. 두 조건에서 ADAPT가 일반 정책보다 높고 Teacher와의 점수 차이가 작다는 뜻이다. 모든 관절의 전체 평균이나 생존율로 해석하지 않는다.

## 4. Onset은 one-shot과 다른 용어

**Onset은 어떤 현상이 시작되는 시점**이다. `failure onset`은 고장이 발생하기 시작하는 순간이다.

- Fixed failure: episode 시작부터 고장이 적용된다.
- Onset failure: 정상 상태로 걷다가 특정 시점에 고장이 적용된다.
- 현재 평가: 시작 후2초 /5초 /10초 중 정해진 조건에서 한 관절의 출력을 저하시킨 뒤20초를 관찰한다. 고장은 적용된 뒤 지속된다.
- 현재 회복 시간은 고장 발생 시점을 기준으로 잰다.

예를 들어5초 onset은0–5초 정상 보행 →5초 고장 발생 →5–25초 고장 이후 평가다.

One-shot은 문맥에 따라 한 개 예제로 학습하거나 한 번의 시도로 해결하는 방식을 뜻한다. Onset이라는 용어 자체는 학습 횟수와 관계없다. 현재 한 episode에 고장 하나를 발생시키도록 정했더라도, 그것 때문에 onset을one-shot이라고 부르지 않는다.

평가 중에는 신경망 가중치를 새로 학습하지 않는다. 관측 이력이 바뀌면서 Student latent와 행동이 달라지는 방식으로 대응한다.

## 5. 다른 논문과 성능적으로 비교하면

기존 수치의 관점에서 비교할 수 있는 부분은 다음과 같다. 논문별 고장·지형·시간 및 성공 조건을 통일한 재평가 결과는 아니다.

| 평가 축 | Saving the Limping | RMA | JT71500 |
|---|---|---|---|
| 평균 생존 시간의 정규화 | 56.5% (20초 기준11.30초) | 85% (논문 최대episode시간 기준) | 97.19% (20초 기준19.44초) |
| 성공/완주 비율 | 위56.5%를 완주율로 사용할 수 없음 | 과제 성공률73.5% | 고장 후20초 완주율95.01% |
| 속도 추종 | 명령0.5, 고장 후 평균0.47m/s | 본 비교표에 대응하는 속도 RMSE 없음 | 명령0.5, episode 평균 전진 RMSE0.0961m/s |
| 고장/변화 | 단일 관절 잠금 | 물리조건 변화 | 단일 관절 출력 저하 |

**보고된 생존 수치 자체는 JT71500이 더 높다.** 따라서 우리 조건에서 높은 생존 수준을 달성했다고 설명할 수 있다. 하지만 관절 잠금과 토크 출력 저하는 다른 고장이고, RMA의 과제 성공과 우리20초 완주도 다른 조건이다. 이 숫자만으로 다른 알고리즘보다 우수하다는 결론은 낼 수 없다.

속도는 특히 평균과 RMSE를 섞지 않아야 한다. Saving the Limping의 평균0.47은 목표대비94%이고, 우리 RMSE0.0961은 명령크기의19.2%지만, 평균 편향과 RMS오차는 같은 통계가 아니므로6%와19.2%를 직접 우열 비교하지 않는다.

현재 가장 강한 성능 비교 근거는 프로젝트 내 같은 조건의 TF43000 ↔ JT71500이다.

| Onset 결과 | TF43000 | JT71500 | 해석 |
|---|---:|---:|---|
| 20초 완주율 | 94.20% | 95.01% | Student의 생존 수준은 유사; 작은 차이를 통계적 우월성으로 주장하지 않음 |
| 안정 회복률 | 90.09% | 81.66% | Student가8.44%p 낮음 |
| 전진 속도 RMSE | 0.0770m/s | 0.0961m/s | Student의 추종 오차가 더 큼 |

발표에서 주장할 수 있는 수준:

> 출력 저하 조건에서 privileged 고장 정보를 직접 받지 않는 Student가 Teacher와 유사한 생존 수준을 달성했습니다. 다만 안정 회복과 속도 추종에는 격차가 남아 있습니다. 선행논문의 결과는 참고 수준으로 비교했으며 동일 고장 환경에서의 분리학습 대비 공동학습 효과 검증이 추가로 필요합니다.

## 근거

- [Saving the Limping 원문 §IV-A/Table I](../papers/main/2023-liu-saving-the-limping/paper.md), [공개 논문](https://arxiv.org/abs/2210.00474).
- [RMA 원문 §IV-B/Table II](../papers/main/2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md), [공개 논문](https://arxiv.org/abs/2107.04034).
- [현재 encoder](/home/jihun/legged_gym/legged_gym/learning/joint_teacher_student_actor_critic.py:19).
- [A1 action scale](/home/jihun/legged_gym/legged_gym/envs/a1/a1_config.py:54).
- [P5 비교 결과](../results/p5-tf43000-vs-jt71500.md), [생존 시간 재집계](assets/v10_evidence.json).

PPTX는 편집 가능한 공식 수식을 보존한다. PDF는 수식 호환용 검수 사본이며 Microsoft PowerPoint 자체에서의 렌더링 확인을 의미하지 않는다.
