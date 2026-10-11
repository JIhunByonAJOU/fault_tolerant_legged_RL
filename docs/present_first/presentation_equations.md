# v12 발표자료 수식 해설 — 페이지별 의미·변수·단위·발표 설명

- 기준 파일: [융캡2 초안 발표 최종본/융합캡스톤디자인2_1조_초안발표_최종본.pptx](융캡2 초안 발표 최종본/융합캡스톤디자인2_1조_초안발표_최종본.pptx), 총 43페이지.
- 작성 목적: 발표자가 수식을 읽고, 각 기호와 구현상의 의미를 설명할 수 있도록 정리한다.
- `슬라이드 수식`은 실제 PPT의 식을 LaTeX로 옮긴 것이다. `보충 수식`은 이해를 돕기 위해 추가한 설명이며 PPT에 모두 있는 것은 아니다.
- 수식이 없는 페이지도 해당 페이지에서 전달할 내용과 관련 수식을 설명한다. PPT 파일 자체는 수정하지 않았다.
- 근거: 현재 PPT의 Office Math, 수식 미리보기 manifest, 환경·학습·평가 코드, 해당 학습의 저장 설정. 새 학습이나 시뮬레이션 없이 확인했다.

## 읽기 전에: 공통 기호와 혼동하기 쉬운 표기

| 기호 | 읽는 법 | 의미·주의 |
|---|---|---|
| \(t\) | 티 | 환경의 시간 인덱스. 물리 시간은 보통 \(t\Delta t\)로 환산 |
| \(k\) | 케이 | 학습 iteration 번호. 환경 step과 다름 |
| \(j\) | 제이 | 관절 번호, 총 12개 |
| \(i\) | 아이 | 문맥에 따라 보상 항 번호 또는 평가 episode 번호 |
| \(\pi\) | 파이, pi | 행동 정책 |
| \(\phi,\varphi\) | 파이, phi | Student encoder. 두 표기는 여기서 같은 역할 |
| \(\mu\) | 뮤 | Teacher encoder |
| \(\xi\) | 크사이, xi | Teacher에 입력하는 물리정보 벡터 |
| \(\tau\) | 타우 | 관절 토크, N·m |
| \(\omega\) | 오메가 | 각속도, rad/s |
| \(\gamma\) | 감마 | 미래 보상의 할인율. 학습률이 아님 |
| \(\rho_0\) | 로 제로 | 초기 상태의 확률분포. 0.99라는 숫자가 아님 |
| \(\alpha\) | 알파 | Actor 입력 latent의 Student 혼합 비중 |
| \(\beta\) | 베타 | latent 모방 손실의 가중치 |
| \(\mathbb E\) | 기댓값 | 확률적 결과에 대한 평균. 코드에서는 표본 평균으로 추정 |
| \(\mathbf1\{C\}\) | 지시함수 | 조건 \(C\)가 참이면 1, 아니면 0 |
| \(\|x\|_2\) | 엘투 노름 | 벡터 길이: \(\sqrt{\sum_l x_l^2}\) |
| \(\|x\|_2^2\) | 엘투 노름 제곱 | 성분의 제곱합: \(\sum_l x_l^2\) |
| \(\operatorname{clip}(x,l,u)\) | 클립 | 값을 하한 \(l\), 상한 \(u\) 안으로 제한 |
| 위첨자 \(T,S\) | 티처, 스튜던트 | 이 문서의 latent에서는 전치·거듭제곱이 아닌 경로 구분 |
| 위첨자 cmd/des | 커맨드/디자이어드 | 명령값/목표값 구분. 거듭제곱이 아님 |

**먼저 기억할 구분:** 환경 보상 \(r_t\)는 행동의 결과에 주는 점수이고, 학습 손실 \(L\)은 신경망 가중치를 갱신할 때 최소화하는 값이다. 생존율·회복률·RMSE는 학습 후 성능을 측정하는 평가 지표다.

---

## 01페이지 · 초안 발표

별도 수식은 없다. 연구 범위는 **Isaac Gym 시뮬레이션에서 단일 관절의 모터 출력 저하에 적응하는 보행 제어**다.

**발표 설명:** “고장 정보를 직접 입력받지 않는 Student가 움직임의 이력을 이용해 고장 후에도 보행을 유지하고 명령 속도를 회복하는지 평가합니다.”

## 02페이지 · 연구 개요

별도 수식은 없다. 목표를 다음 세 가지로 구분한다.

- 생존: 고장 상태에서 평가 시간을 끝까지 버티는가.
- 추종: 실제 속도가 명령 속도에 얼마나 가까운가.
- 회복: 고장 후 허용 오차 범위에 들어와 1초 연속 유지하는가.

보상·정책·평가 지표는 각각 12–13, 15–19, 21페이지에서 수식으로 정의한다. 높은 생존율만으로 높은 추종 성능을 주장하지 않는다.

## 03페이지 · 4족 보행 로봇의 구조

**보충 수식**

$$
n_a=4\ \text{다리}\times3\ \text{관절/다리}=12.
$$

\(n_a\)는 구동 관절 수이자 행동 차원이다. 각 다리에 hip, thigh, calf가 하나씩 있다. 몸체의 공간 이동·회전 자유도까지 모두 합쳐 로봇 전체 자유도가 12라고 해석하면 안 된다.

**발표 설명:** “정책은 네 다리의 총 12개 관절에 대한 목표 각도 보정량을 동시에 출력합니다.”

## 04페이지 · 액추에이터 고장의 분류

별도 수식은 없다. 다음 차이를 알아야 11페이지를 설명할 수 있다.

- Joint locking: 관절의 가동 범위를 특정 각도 주변으로 제한한다.
- Output degradation: 관절 가동 범위는 유지하면서 모터 토크를 줄인다.
- 완전 출력 상실: 열화율 \(d=1\), 적용 모터 토크가 0이다. 기계적 잠금과 다르다.

출력이 0이어도 중력·지면 반력·다른 링크의 운동 때문에 해당 관절이 움직일 수 있다. 모든 기계적 저항까지 없어진다는 뜻은 아니다.

## 05페이지 · 연구 문제와 목표

**보충 수식**

$$
a_t=\pi(o_t,z_t^S),\qquad z_t^S=\phi(h_t).
$$

\(a_t\): 12차원 행동, \(o_t\): 현재 관측, \(h_t\): 최근 관측 이력, \(z_t^S\): Student가 추정한 latent다. 이 식은 평가 시 결정론적 행동 출력의 개념적 표기다. 학습에서는 정책의 확률분포에서 행동을 표본 추출한다.

**발표 설명:** “고장 위치와 정도를 정답으로 알려주는 대신, 이력에서 고장의 영향을 반영하는 특징을 추정해 관절 명령을 조정합니다.”

## 06페이지 · 선행연구와 공동학습의 근거

별도 수식은 없다. 각 수식의 출처를 구분한다.

| 요소 | 근거 |
|---|---|
| 이력으로 환경 특징을 추정 | RMA 계열의 적응 구조 |
| Teacher·Student latent 혼합과 공동학습 | Saving the Limping |
| \(\tau_{\mathrm{applied}}=(1-d)\tau_{\mathrm{desired}}\) | ADAPT의 출력 저하 모델 |
| 현재 보상 항과 계수 | Rudin 등의 legged_gym 공개 구현 중 A1 설정 |

Isaac Gym은 물리 시뮬레이터이며, 그 자체가 이 프로젝트의 보상함수를 자동 제공하는 것은 아니다. RMA를 4족 보행 Teacher–Student의 최초 연구라고 단정하지 않는다.

## 07페이지 · 프로젝트 변형과 연구 기여

별도 수식은 없다. **계승한 수학적 구조와 프로젝트에서 정한 수치**를 분리해서 말한다.

- 계승: 토크 비율 감소, latent 혼합, Teacher–Student 공동학습.
- 프로젝트 설정: 관측 235D, Teacher 물리정보 45D, 50×48 이력, 10,000 iteration 전환, 평가 속도 0.5 m/s, 1초 회복 기준.

현재 결과만으로 공동학습이 분리학습보다 우수하다고 말할 수는 없다. 같은 조건의 분리학습 Student 기준선이 별도로 필요하다.

## 08페이지 · Isaac Gym 환경

슬라이드의 rollout 계산은 다음과 같이 전체 곱셈을 포함해 읽는다.

$$
N_{\mathrm{env}}=4096,\qquad H_{\mathrm{rollout}}=24,
$$

$$
B=N_{\mathrm{env}}H_{\mathrm{rollout}}=4096\times24=98{,}304
\quad\text{transitions/iteration}.
$$

| 변수 | 의미 |
|---|---|
| \(N_{\mathrm{env}}\) | 병렬 시뮬레이션 환경 수 |
| \(H_{\mathrm{rollout}}\) | 환경 하나에서 새로 수집하는 정책 step 수 |
| \(B\) | 한 iteration의 새 transition 수 |

**주의:** 슬라이드의 `24 steps/env = 98,304 transitions/iteration`은 병렬 환경 수가 생략된 축약 표기다. 24 자체가 98,304와 같다는 뜻이 아니다.

$$
\Delta t_{\mathrm{physics}}=1/200=0.005\ \mathrm{s},\qquad
\Delta t_{\mathrm{policy}}=1/50=0.02\ \mathrm{s}.
$$

정책 행동 하나를 유지하는 동안 물리 시뮬레이션은 4번 진행한다. 새 rollout은 환경당 \(24\times0.02=0.48\)초 구간을 수집하며, episode 전체 20초와 다른 단위다.

## 09페이지 · 제어 문제의 정식화

**슬라이드 수식**

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma,\rho_0),\qquad\gamma=0.99.
$$

| 기호 | 의미 | 현재 문제 |
|---|---|---|
| \(\mathcal S\) | 상태 공간 | 로봇 자세·속도·접촉·지형·물리계수·명령·고장 관련 상태 등 |
| \(\mathcal A\) | 행동 공간 | 12개 관절각 보정 행동 |
| \(P\) | 상태 전이 분포 | PD 제어, 출력 저하, 물리 시뮬레이션으로 다음 상태 생성 |
| \(R\) | 보상 규칙 | 속도 추종·안정성·제어 비용의 가중합 |
| \(\gamma\) | 할인율 | 한 정책 step 뒤 보상에 곱하는 비율 |
| \(\rho_0\) | 초기 상태 분포 | episode 초기 로봇·환경 조건의 분포 |

$$
\pi^*=\arg\max_\pi\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
$$

- \(\pi\): 관측과 latent를 받아 행동을 정하는 정책.
- \(\pi^*\): 기대 할인 누적 보상을 가장 크게 하는 정책.
- \(T\): episode의 유효 step 수. 20초 시간 제한만 적용되면 약 1,000 policy steps이고, 조기 종료 시 더 짧다.
- \(r_t\): 그 step의 스칼라 보상.
- \(\gamma^t\): 먼 미래의 보상에 적용하는 가중치. \(0.99^{100}\approx0.366\)이다.

**감마는 학습률이 아니다.** 학습률은 가중치 갱신 크기를 정하며, 감마는 학습 목표에서 미래 보상을 얼마나 중시할지 정한다.

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),\qquad o_t\sim O(\cdot\mid s_t).
$$

\(s_t\)는 전체 상태, \(o_t\)는 정책에 제공하는 관측, \(O\)는 관측 생성 규칙이다. 물리적으로 전체 상태를 정의하면 MDP로 모델링할 수 있지만, Student는 고장·물리계수의 정답을 직접 받지 않으므로 **부분 관측 제어 문제**다. \(O\)까지 포함한 엄밀한 POMDP 정의는 슬라이드 MDP 튜플보다 확장된 표현이다.

마르코프 상태를 엄밀히 구성할 때는 이전 행동처럼 보상에 필요한 기억, 고장 발생 일정과 episode clock 등 미래 전이에 필요한 변수도 포함한다. 유한 이력을 넣는다고 완전한 상태 관측이 보장되는 것은 아니다.

**발표 설명:** “물리 상태에 따라 로봇이 전이하는 환경을 정의하고, Student가 관측 가능한 정보와 이력을 이용해 할인 누적 보상을 최대화하도록 학습합니다.”

## 10페이지 · 관측 및 행동 공간

**관측의 물리적 의미를 설명하는 보충 식**

$$
o_t=\operatorname{concat}\left(
\mathbf v_{b,t},\boldsymbol\omega_{b,t},\mathbf g_{b,t},\mathbf c_t,
\mathbf q_t-\mathbf q_0,\dot{\mathbf q}_t,\mathbf a_{t-1},\mathbf m_t
\right)\in\mathbb R^{235}.
$$

실제 코드에서는 성분별 scaling·clipping 및 설정에 따른 noise가 적용된다. 위 식은 원시 물리량 기준의 구성 설명이다.

| 성분 | 차원 | 의미 | 원시 물리 단위 |
|---|---:|---|---|
| \(\mathbf v_b\) | 3 | 몸체 좌표계 선속도 | m/s |
| \(\boldsymbol\omega_b\) | 3 | 몸체 각속도 | rad/s |
| \(\mathbf g_b\) | 3 | 몸체 좌표계에 투영한 중력 방향 | 무차원 |
| \(\mathbf c_t\) | 3 | 전진·횡속도·yaw 각속도 명령 | m/s, m/s, rad/s |
| \(\mathbf q-\mathbf q_0\) | 12 | 기본 자세 대비 관절각 차이 | rad |
| \(\dot{\mathbf q}\) | 12 | 관절 각속도 | rad/s |
| \(\mathbf a_{t-1}\) | 12 | 직전 행동 | 무차원 |
| \(\mathbf m_t\) | 187 | 주변 지형 상대 높이 정보 | m |

$$
3+3+3+3+12+12+12=48,\qquad48+187=235.
$$

지형 187개는 \(17\times11\)개의 수평 샘플 위치다. x 방향 −0.8~0.8 m를 0.1 m 간격으로 잡으면 17점, y 방향 −0.5~0.5 m는 11점이다. 17과 11은 높이 개수가 아니라 가로·세로 방향의 위치 개수다.

**슬라이드 행동 수식**

$$
\mathbf q_t^{\mathrm{des}}=\mathbf q^{\mathrm{default}}+0.25\mathbf a_t,
\qquad\mathbf a_t\in\mathbb R^{12}.
$$

\(\mathbf a_t\)는 무차원 행동이며 0.25는 행동 1단위당 0.25 rad의 각도 보정량을 주는 action scale이다. \(\mathbf q^{\mathrm{default}}\)는 고정된 기준 자세이며 직전 목표각이 아니다. Actor가 관절 속도를 직접 출력하는 식이 아니다.

예: 기본각 0.8 rad, 행동 0.4 → 목표각 \(0.8+0.25\times0.4=0.9\) rad.

도→라디안 변환은 \(q_{\mathrm{rad}}=q_{\mathrm{deg}}\pi/180\)이다. 0.25와 관계없다. 또한 \(\mathbb R^{12}\)는 차원을 의미하며 행동이 자동으로 [−1,1]에 제한된다는 뜻은 아니다.

## 11페이지 · 출력 저하 동역학

**슬라이드 수식 1: PD 명령 토크**

$$
\tau_{j,t}^{\mathrm{cmd}}=K_{p,j}(q_{j,t}^{\mathrm{des}}-q_{j,t})-K_{d,j}\dot q_{j,t}.
$$

| 변수 | 의미 | 단위 |
|---|---|---|
| \(q^{\mathrm{des}}_{j,t},q_{j,t}\) | 목표·현재 관절각 | rad |
| \(\dot q_{j,t}\) | 현재 관절 각속도 | rad/s |
| \(K_{p,j}\) | 각도 오차에 대한 비례 게인 | N·m/rad |
| \(K_{d,j}\) | 관절 움직임의 감쇠 게인 | N·m·s/rad |
| \(\tau^{\mathrm{cmd}}_{j,t}\) | 계산한 명령 토크 | N·m |

첫 항은 목표각으로 움직이게 하고, 두 번째 항은 과도한 움직임을 줄인다. 목표 관절 속도를 0으로 두므로 D항이 \(-K_d\dot q\)가 된다. 실제 코드에서는 PD 게인에 무작위화 배율도 적용한다.

**슬라이드 수식 2: 출력 저하**

$$
\tau_{j,t}^{\mathrm{applied}}=(1-d_{j,t})\tau_{j,t}^{\mathrm{cmd}}.
$$

\(d_{j,t}\)는 무차원 열화율이다. 0이면 정상, 0.8이면 20% 출력 잔존, 1이면 해당 모터 토크가 0이다.

**슬라이드 수식 3: 관절과 발생 시점 지정**

$$
d_{j,t}=d\,\mathbf1\{j=J_f,\ t\geq T_f\},\qquad d\in[0,1].
$$

\(J_f\)는 고장 관절, \(T_f\)는 고장 발생 시점이다. 선택된 관절이며 고장 시점 이후라는 두 조건을 동시에 만족할 때만 \(d\)를 적용한다. 이 식에서 \(t,T_f\)는 같은 단위로 비교해야 한다. 코드에서는 초를 정책 step 번호로 환산한다.

예: 왼쪽 뒤 hip에 5초부터 \(d=0.8\)을 적용하면, 그 전에는 정상이고 그 후에는 해당 관절의 출력만 20%로 줄어든다. 다른 관절의 명령도 정책 반응에 따라 달라질 수 있지만, 고장 계수 자체는 한 관절에만 적용한다.

**실제 구현 보충:** PD 토크에 nominal motor-strength 배율 \(m_j^{\mathrm{nom}}\)과 \((1-d_{j,t})\)를 곱한 뒤 토크 한계로 clip한다.

$$
\tau_{j,t}^{\mathrm{sim}}=
\operatorname{clip}\left(m_j^{\mathrm{nom}}(1-d_{j,t})\tau_{j,t}^{\mathrm{PD}},
-\tau_j^{\max},\tau_j^{\max}\right).
$$

\(m_j^{\mathrm{nom}}\)은 무차원 정상 출력 배율, \(\tau_j^{\max}\)는 관절 토크 한계다. 슬라이드는 고장 효과를 설명하기 위해 이 추가 요소를 생략했다. 포화가 있으면 “정상일 때 이미 clip된 토크를 단순히 20%로 줄인다”와는 계산 순서가 다르다.

**발표 설명:** “정책의 목표 관절각을 PD 토크로 변환하고, 선택된 관절에 고장 시점 이후 출력 저하율을 적용합니다. 그 토크로 물리 시뮬레이션의 다음 상태가 결정됩니다.”

## 12페이지 · 보상함수의 구성

**슬라이드 수식**

$$
r_t=\max\left(0,\Delta t\sum_{i=1}^{10}w_i f_{i,t}\right),
\qquad\Delta t=0.02\ \mathrm{s}.
$$

\(f_{i,t}\)는 i번째 원시 보상/비용 항, \(w_i\)는 계수, \(r_t\)는 최종 step 보상이다. 비용 함수 자체는 양수여도 음의 계수를 곱하면 페널티가 된다. 전체 합을 만든 뒤 음수이면 0으로 자른다. 현재 종료 reward 계수는 0이다.

| 원시 항 | 주요 정의 | 계수 |
|---|---|---:|
| 평면 속도 추종 | 13페이지 지수 함수 | 1.0 |
| yaw 추종 | 13페이지 지수 함수 | 0.5 |
| 수직 속도 | \(v_z^2\) | −2.0 |
| roll·pitch 각속도 | \(\omega_x^2+\omega_y^2\) | −0.05 |
| 토크 | \(\sum_j\tau_j^2\) | \(-2\times10^{-4}\) |
| 관절 가속도 | \(\sum_j[(\dot q_{j,t}-\dot q_{j,t-1})/\Delta t]^2\) | \(-2.5\times10^{-7}\) |
| 행동 변화 | \(\sum_j(a_{j,t}-a_{j,t-1})^2\) | −0.01 |
| thigh·calf 충돌 | 임계 접촉력을 넘은 body 개수 | −1.0 |
| 관절 위치 한계 | soft limit 밖의 각도량 합 | −10.0 |
| 발 체공 시간 | 첫 착지 시 체공 시간 − 0.5초 | 1.0 |

물리 단위가 다른 항들을 가중치로 조절해 하나의 학습 점수로 합한다. 계수 숫자가 크다는 이유만으로 그 항의 실제 기여가 항상 크다고 판단할 수는 없다. 원시 항의 크기까지 봐야 한다.

현재 항과 계수의 직접 출처는 **Rudin 등의 legged_gym 공개 코드 A1 설정**이다. Saving the Limping의 보상 전체를 정확히 복제했다고 말하지 않는다.

**발표 설명:** “보상은 하나의 숫자지만, 추종을 장려하고 불안정한 움직임과 불필요한 제어를 억제하는 10개 항의 가중합입니다.”

## 13페이지 · 추종 목표의 보상 표현

**슬라이드 수식**

$$
f_{\mathrm{lin},t}=\exp\left(-\frac{\|\mathbf v_{xy,t}^{\mathrm{cmd}}-\mathbf v_{xy,t}\|_2^2}{0.25}\right),
$$

$$
f_{\mathrm{yaw},t}=\exp\left(-\frac{(\omega_{z,t}^{\mathrm{cmd}}-\omega_{z,t})^2}{0.25}\right).
$$

\(\mathbf v_{xy}=[v_x,v_y]\)는 몸체 좌표계 평면 속도(m/s), \(\omega_z\)는 몸체 z축 각속도(rad/s), cmd는 명령값이다. 분자의 속도 오차 제곱은 다음과 같다.

$$
\|\mathbf v_{xy}^{\mathrm{cmd}}-\mathbf v_{xy}\|_2^2
=(v_x^{\mathrm{cmd}}-v_x)^2+(v_y^{\mathrm{cmd}}-v_y)^2.
$$

오차 0 → \(\exp(0)=1\). 오차가 커질수록 0에 가까워진다. 0.25는 오차 민감도를 정하는 `tracking_sigma` 설정이며, 코드에서는 그 값으로 직접 나눈다. 0.25를 다시 제곱하는 것이 아니다. 물리 단위 관점에서는 각각 속도제곱·각속도제곱에 대응하는 폭으로 해석해야 지수 안이 무차원이 된다. 두 식은 같은 숫자 설정을 사용한다.

횡속도 오차 0, 전진 명령 0.5 m/s일 때:

| 실제 전진 속도 | \(f_{\mathrm{lin}}\) | step 보상 기여 \(0.02\times1.0\times f_{\mathrm{lin}}\) |
|---:|---:|---:|
| 0.5 m/s | 1.000 | 0.0200 |
| 0.4 m/s | 0.961 | 0.0192 |
| 0.0 m/s | 0.368 | 0.0074 |

이 보상 함수의 0.25와 행동 스케일 0.25는 우연히 숫자가 같을 뿐 역할이 다르다. 또한 회복 판정의 ±0.1 m/s는 이 함수의 파라미터가 아니라 별도의 평가 기준이다.

**발표 설명:** “추종 오차의 제곱을 지수 보상으로 바꿔, 명령에 가까울수록 높은 점수를 주고 오차가 커질수록 연속적으로 낮춥니다.”

## 14페이지 · Teacher–Student의 학습 목적

별도 수식은 없다. 15–19페이지를 이해하기 위한 목적 설명이다.

Teacher는 물리정보를 이용해 행동에 유용한 특징을 만든다. Student는 그 물리정보를 직접 받지 않고 관측 이력으로 특징을 추정한다. 추정 특징에는 오차가 있으므로 Actor도 함께 갱신하여 그 입력으로 보행할 수 있게 한다.

**주의:** 평가 중 매 step 신경망을 재학습한다는 뜻이 아니다. 가중치는 고정된 채, 이력에 따라 latent와 행동이 달라지는 방식으로 적응한다.

## 15페이지 · Teacher–Student 정보 비대칭

**슬라이드 수식**

$$
z_t^T=\mu(\xi_t)\in\mathbb R^8,\qquad z_t^S=\varphi(h_t)\in\mathbb R^8.
$$

| 변수 | 의미 |
|---|---|
| \(\xi_t\in\mathbb R^{45}\) | Teacher 물리정보 입력 |
| \(\mu\) | 학습 가능한 Teacher encoder |
| \(h_t\in\mathbb R^{50\times48}\) | 최근 50프레임의 관측·행동 이력 |
| \(\varphi\) | 학습 가능한 Student encoder |
| \(z_t^T,z_t^S\) | 각각의 8차원 특징 벡터, latent |

Teacher 45차원은 다음처럼 구성된다.

$$
1_{\mathrm{friction}}+1_{\mathrm{payload}}+12_{\mathrm{motor}}
+12_{K_p}+12_{K_d}+3_{v}+3_{\omega}+1_{\mathrm{failure}}=45.
$$

마찰계수·모터/게인 배율·고장 flag는 무차원, 추가 질량은 kg, 선속도는 m/s, 각속도는 rad/s다. 고장 flag는 현재 코드에서 고장 유무를 나타내며 관절 번호 자체가 아니다. 관절별 motor strength 벡터가 출력 차이를 제공한다.

Student는 각 프레임의 첫 48개 관측 성분을 받는다. 지형 187D는 history encoder에 쌓지 않지만 현재 관측 235D를 통해 Actor에 제공한다. 50프레임은 50 Hz에서 관례상 약 1초 이력이며 첫 샘플과 마지막 샘플의 시각 차이는 49×0.02=0.98초다.

8개 latent 성분을 “마찰·질량 등 물리량 8개”로 각각 지정한 것은 아니다. 신경망이 학습한 표현이므로 성분별 물리 의미는 검증 없이 부여하지 않는다.

## 16페이지 · Teacher–Student 구조

**보충 수식**

$$
x_t=\operatorname{concat}(o_t,z'_t)\in\mathbb R^{243},\qquad235+8=243.
$$

\(z'_t\)는 그 단계에서 선택한 latent다. Teacher 정책에서는 Teacher latent, 혼합 학습에서는 혼합 latent, Student 평가에서는 Student latent다.

학습 시 개념적으로:

$$
a_t\sim\pi_\theta(\cdot\mid x_t),\qquad V_\psi(x_t)\in\mathbb R.
$$

\(\theta\)는 Actor 가중치, \(\psi\)는 Critic 가중치, \(V_\psi\)는 앞으로 받을 할인 누적 보상의 예측값이다. 현재 구현의 Actor는 가우시안 정책의 평균을 만들고 행동 표준편차도 학습한다. 평가에서는 평균 행동을 사용한다.

**Actor는 하나이며 입력은 항상 243D다.** 혼합 시 Teacher 8D와 Student 8D를 이어 붙여 16D로 넣는 것이 아니다.

## 17페이지 · 특징 혼합과 Student 전환

**슬라이드 수식**

$$
z_t^{\mathrm{mix}}=(1-\alpha)z_t^T+\alpha z_t^S\in\mathbb R^8.
$$

두 8차원 벡터의 **원소별 가중합**이다. \(\alpha=0\)이면 Teacher만, 0.5이면 각 절반, 1이면 Student만 사용한다. 섞는 대상은 네트워크 가중치나 두 로봇의 행동이 아니라 latent다.

$$
\alpha_k=\operatorname{clip}\left(\frac{k-43{,}000}{10{,}000},0,1\right),\qquad\beta_k=1-\alpha_k.
$$

| 기호 | 의미 |
|---|---|
| \(k\) | global 학습 iteration 번호 |
| 43,000 | 이 공동학습 구간의 시작 번호 |
| 10,000 | Teacher→Student 전환에 쓰는 iteration 수 |
| \(\alpha_k\) | Student latent의 입력 비중 |
| \(\beta_k\) | 19페이지 모방 손실 가중치 |

| iteration | Teacher 비중 | Student 비중 | 모방 가중치 |
|---:|---:|---:|---:|
| 43,000 | 1.0 | 0.0 | 1.0 |
| 45,000 | 0.8 | 0.2 | 0.8 |
| 48,000 | 0.5 | 0.5 | 0.5 |
| 53,000 이후 | 0.0 | 1.0 | 0.0 |

현재 JT71500 계열의 beta floor는 0이다. 다른 ablation 설정의 beta floor와 혼동하지 않는다.

**발표 설명:** “처음부터 미학습 Student 특징만 주면 입력이 급격히 바뀌므로, Teacher 특징에서 Student 특징으로 점진적으로 바꾸며 Actor도 함께 적응시킵니다.”

## 18페이지 · 정책 입력과 관절 명령

**슬라이드 수식**

$$
q_t^{\mathrm{des}}=q^{\mathrm{default}}+0.25a_t,\qquad a_t\in\mathbb R^{12}.
$$

10페이지와 같은 식이다. 이 페이지에서는 전체 연결 순서를 설명한다.

$$
(o_t,h_t)\ \longrightarrow\ z_t^S\ \longrightarrow\ [o_t,z_t^S]
\ \longrightarrow\ a_t\ \longrightarrow\ q_t^{\mathrm{des}}
\ \longrightarrow\tau_t\ \longrightarrow\text{로봇 운동}.
$$

\([o_t,z_t^S]\)는 연결(concatenation)이며, 현재 관측과 latent를 하나의 Actor 입력으로 묶는다. 243D에 Student 8D를 또 더해 251D를 만드는 것이 아니다.

**발표 설명:** “Student가 만든 8차원 특징과 현재 관측을 Actor에 넣고, Actor의 12차원 출력을 기본 관절각의 보정량으로 사용합니다.”

## 19페이지 · 공동학습 목적함수와 갱신 대상

**슬라이드 수식**

$$
L=L_{\mathrm{policy}}+c_VL_{\mathrm{value}}-c_HH(\pi)+\beta L_{\mathrm{adapt}},
\qquad c_V=1,\quad c_H=0.01.
$$

| 항 | 뜻 | 줄이려는 것/장려하는 것 |
|---|---|---|
| \(L_{\mathrm{policy}}\) | PPO 정책 손실 | 기대 누적 보상이 커지는 방향의 행동 선택 |
| \(L_{\mathrm{value}}\) | Critic 손실 | 누적 보상 예측 오차 |
| \(H(\pi)\) | 정책 엔트로피 | 행동 분포가 지나치게 빨리 좁아지는 것을 억제 |
| \(c_V,c_H\) | 각 항의 가중치 | 가치 예측·탐색의 상대적 비중 |
| \(L_{\mathrm{adapt}}\) | Student 특징 모방 손실 | Teacher·Student latent 사이의 거리 |
| \(\beta\) | 모방 손실 가중치 | 초기에는 모방을 강조, 후기에는 감소 |

전체 \(L\)은 최소화한다. 엔트로피 앞이 음수인 이유는 손실을 줄이는 과정에서 엔트로피를 높여 탐색을 장려하기 위해서다. \(L_{\mathrm{policy}}\)는 34페이지 최대화 PPO 목적의 음수에 해당한다.

**모방 손실**

$$
L_{\mathrm{adapt}}=\mathbb E_t\left[\left\|z_t^S-\operatorname{sg}(z_t^T)\right\|_2\right].
$$

- \(\operatorname{sg}\)는 stop-gradient다. 이 손실을 계산할 때 Teacher latent를 정답으로 취급하고 이 경로의 gradient를 막는다.
- 현재 구현은 **L2 거리의 평균**이다. 제곱한 MSE가 아니다.
- \(\mathbb E_t\)는 minibatch 표본 평균으로 구현된다.

성분으로 풀어 쓰면:

$$
\|z_t^S-\operatorname{sg}(z_t^T)\|_2
=\sqrt{\sum_{l=1}^{8}(z_{t,l}^S-z_{t,l}^T)^2}.
$$

여기서 \(l\)은 latent의 성분 번호다. 두 특징이 가까울수록 손실이 작아진다. 8차원 latent의 단위는 명시적인 물리 단위로 정해져 있지 않다.

**어느 신경망이 학습되는가**

| 경로 | 갱신 대상 |
|---|---|
| 모방 손실 | Student encoder. Teacher target은 detach |
| PPO policy/value | Actor·Critic 및 혼합 latent에 참여하는 encoder |
| \(\alpha=0,\beta=1\) 초기 | Student는 모방으로 학습. PPO latent 경로는 Teacher 쪽 |
| \(\alpha=1,\beta=0\) 후기 | Student·Actor·Critic은 PPO 학습, Teacher 경로의 기여는 없음 |

Teacher는 모방 손실에서만 고정 target이며, 공동학습 전체에서 항상 동결되는 것은 아니다. 별도 FrozenTeacher ablation과 구분한다.

**발표 설명:** “PPO로 보행 성능을 높이는 학습과 Student가 Teacher 특징을 따라가는 학습을 함께 수행합니다. 이후 모방 비중을 줄여 Student 특징을 이용한 행동 보상 최적화에 집중합니다.”

## 20페이지 · 학습 절차와 정책 선택

**슬라이드 수식**

$$
4096\times24=98{,}304\quad\text{transitions/iteration},
$$

$$
5\ \text{epochs}\times4\ \text{minibatches/epoch}=20\ \text{optimizer updates/iteration}.
$$

첫 식은 새 경험의 양, 두 번째 식은 그 경험을 재사용해 가중치를 갱신하는 횟수다. 20번 업데이트한다고 새 transition이 20배 만들어지는 것은 아니다.

흐름: Teacher 준비 → Teacher 가중치로 공동학습 초기화 → 새 Student를 모방+PPO 학습 → Student-only 후기 → 별도 평가로 checkpoint 선택.

계산식은 update가 모두 수행되는 정상 iteration 기준이다. 일반적으로 조기 중단·비정상 gradient 처리 등이 있으면 실제 로그의 completed updates도 확인해야 한다.

## 21페이지 · 평가 지표와 선정 근거

### 21-1. 생존율, 20초 완주율

**슬라이드 수식**

$$
S_{\mathrm{complete}}=\frac1N\sum_{i=1}^{N}\mathbf1[T_i=20\ \mathrm{s}].
$$

\(N\)은 전체 평가 시행 수, \(T_i\)는 해당 시행에서 평가 구간 안에 생존한 시간(최대 20초)이다. 비율을 %로 표시하려면 100을 곱한다. 실제 코드는 시간의 부동소수점 동등 비교 대신 종료 step과 종료 예정 step으로 판정한다.

**한 줄:** 생존율은 고장 상태에서 종료 없이 정해진 20초 평가 구간을 끝까지 유지한 시행의 비율이다.

현재 몸체 base 접촉력 norm이 1 N을 넘으면 종료한다. thigh·calf 접촉 페널티는 별도이며 즉시 종료와 동일하지 않다. 속도가 0이어도 종료되지 않으면 생존할 수 있다.

### 21-2. 정규화 평균 생존 시간

$$
S_{\mathrm{TTF}}=\frac{\mathbb E_i[T_i]}{20\ \mathrm{s}}.
$$

이는 완주율과 다르다. 모든 시행이 10초에 종료되면 완주율은 0%, 정규화 평균 생존은 50%다. \(TTF\)는 time to failure의 약자이며, 여기서는 최대 평가시간에서 잘린 생존시간을 사용한다.

### 21-3. 전진 속도 RMSE

$$
\mathrm{RMSE}_{v_x}=\sqrt{\frac1K\sum_{t=1}^{K}(v_{x,t}-v_{x,t}^{\mathrm{cmd}})^2}.
$$

\(K\)는 해당 episode의 고장 후 유효 step 수다. 오차 제곱의 평균에 제곱근을 취하므로 단위는 다시 m/s다. 양·음 오차가 상쇄되지 않는다. 실제 보고값은 이 episode별 RMSE를 다시 평균한 값이며 42페이지에서 구분한다.

### 21-4. 회복률

**슬라이드 판정 조건**

$$
|v_x-0.5|\leq0.1\ \mathrm{m/s},\qquad
|\omega_z|\leq0.2\ \mathrm{rad/s}\quad\text{를 1초 연속 만족}.
$$

속도 명령은 0.5 m/s, yaw 명령은 0이다. 즉 실제 전진 속도 0.4~0.6 m/s이며 회전 각속도 약 ±11.5°/s 안에 있어야 한다. 50 Hz에서는 50개 연속 판정 step이다.

**보충 정의**

$$
C_{i,t}=\mathbf1\{|v_{x,i,t}-0.5|\leq0.1,\ |\omega_{z,i,t}|\leq0.2\},
$$

$$
R_i=\mathbf1\left\{\exists t_0:\ \prod_{\ell=0}^{49}C_{i,t_0+\ell}=1\right\},
\qquad S_{\mathrm{recovery}}=\frac1N\sum_i R_i.
$$

여기서 \(t_0\)부터의 50개 시점은 모두 고장 후 유효 평가 구간에 있어야 한다. \(R_i\)는 회복 성공 여부다. 이 표기에서 \(R_i\)는 MDP 보상 규칙 \(R\)과 다른 기호다.

**한 줄:** 회복률은 고장 후 전진 속도 오차 ±0.1 m/s 및 yaw 각속도 ±0.2 rad/s 이내를 1초 연속 유지한 시행의 비율이다.

roll·pitch 자세각, 횡속도, 몸체 높이는 이 회복 판정에 직접 포함되지 않는다. 한 번 기준을 달성한 뒤 넘어져도 회복 성공 기록은 유지된다. 따라서 “평가가 끝날 때까지 정상 보행”과 다르다.

회복 시간은 **처음 성공한 1초 구간의 시작 시점 − 고장 시점**으로 기록한다. 성공 여부를 확정하는 시점은 그 구간 끝이지만, 기록값은 구간 시작 기준이다.

## 22페이지 · 평가 프로토콜

**보충 계산**

$$
N_{\mathrm{fixed}}=12\times6\times3\times16=3456,
$$

$$
N_{\mathrm{onset}}=12\times5\times3\times3\times16=8640.
$$

각 인자는 순서대로 관절 수, 열화율 수, (onset에서는 발생 시점 수), 평가 시드 수, 조건당 반복 수다.

- Fixed: 시작부터 고장. 열화율 0, 0.2, 0.4, 0.6, 0.8, 1.0. 평가 20초.
- Onset: 2·5·10초 정상 보행 후 고장. 열화율 0.2~1.0. 고장 후 20초 평가.
- 명령: 전진 0.5 m/s, 횡속도 0, yaw 각속도 0.
- 평가 시드: 1, 2, 3. 동일 체크포인트를 서로 다른 난수 시작값에서 평가한다. 독립 학습 3회가 아니다.

시드는 지형만이 아니라 초기상태·활성 물리 무작위화 등에도 영향을 준다. 명령·고장 관절·열화율·고장 시점은 평가 격자로 정한 조건이다. 같은 시드만으로 Teacher/Student의 모든 물리 난수 표본이 일대일로 같다고 보장할 수는 없다.

**발표 설명:** “고장을 처음부터 적용하는 시험과 주행 중 발생시키는 시험을 나눠, 지정한 모든 관절·열화율 조합을 세 평가 시드에서 반복했습니다.”

## 23페이지 · TF43000 학습 결과

별도 수식은 없다. 그래프 값은 시뮬레이션 성능 지표와 구분한다.

- Reward: 학습 과정의 보상 집계. 여러 보상 항을 합친 값이므로 그 숫자 자체가 m/s나 생존율은 아니다.
- Episode length: 시간 제한 1,000 policy steps에 얼마나 가까워지는지 확인한다.
- Terrain level: 지형 curriculum에서 도달한 난도 지표이며 물리 단위가 아니다.

**보충 환산**

$$
980.6\ \text{steps}\times0.02\ \mathrm{s/step}=19.612\ \mathrm{s}.
$$

이는 해당 학습 로그의 평균 episode 길이를 시간으로 읽는 예시다. 평가 완주율 98.06%라는 뜻은 아니다.

## 24페이지 · JT71500 학습 결과

별도 수식은 없다. 17페이지 schedule에서:

$$
\alpha_{71500}=1,\qquad\beta_{71500}=0.
$$

따라서 이 구간의 행동은 Student 특징으로 생성된다. 71,500은 checkpoint의 global iteration 번호다. 학습 그래프가 안정됐다는 것과 새 고장 조건에 잘 일반화한다는 것은 별개의 주장이다.

988.6 policy steps는 평균 약 19.772초다. 이것 역시 평가 완주율로 바로 환산하지 않는다.

## 25페이지 · 공동학습 초기 구간

**보충 계산**

$$
\alpha_{45000}=\frac{45000-43000}{10000}=0.2.
$$

초기 reward 상승에는 Teacher latent 80~100%가 기여한다. 따라서 “45,000에서 Student 단독 학습이 거의 끝났다”라고 해석하면 안 된다.

별도 Student-only onset 평가에서 JT45000 회복률 38.82%, JT71500 회복률 81.66%를 비교한다. 학습 중 혼합 정책의 보상과 평가 Student 단독 성능은 입력 경로가 다르다.

## 26페이지 · 최종 정량 결과

별도 수식은 없다. 21페이지 정의로 계산한 Teacher/Student 지표를 비교한다.

**보충 식: percentage-point 차이**

$$
\Delta_{\mathrm{pp}}=100(S_{\mathrm{Student}}-S_{\mathrm{Teacher}}).
$$

95.01%−94.20%=0.81%p는 비율의 차이다. 상대 개선율 \((95.01-94.20)/94.20\times100\)과 다르다. 표의 차이는 반올림 전 수치로 계산되어 표시된 두 숫자의 단순 차이와 0.01%p 정도 다를 수 있다.

Onset recovery는 전체 열화율의 회복률, 완전 상실 회복률은 같은 판정 기준을 \(d=1\) 시행에만 적용한 값이다. 둘의 판정식이 다른 것이 아니다.

**발표 설명:** “Student는 Teacher와 비슷한 생존 수준을 유지하지만, 회복률과 속도·yaw 추종 오차에는 격차가 남아 있습니다.”

## 27페이지 · 선행연구 결과와 비교 범위

**보충 환산**

$$
0.565\times20\ \mathrm{s}=11.30\ \mathrm{s},\qquad
\frac{19.44}{20}\approx0.972.
$$

Saving the Limping의 56.5%는 정규화 평균 생존 시간이고, 우리 95.01%는 20초 완주율이다. 둘을 같은 지표로 바로 비교하지 않는다. 비교 축을 맞추려면 우리 평균 생존시간 약 19.44초/정규화 97.19%와 함께 표시한다.

그래도 고장 모델·지형·명령 조건이 달라 알고리즘 우월성의 증거가 되지는 않는다. 평균 속도 0.47 m/s도 RMSE 0.03 m/s를 의미하지 않는다. ADAPT reward는 구성·집계가 달라 우리 reward와 원점수로 비교할 수 없다.

## 28페이지 · 결과 영상

별도 수식은 없다. 영상은 명령과 실제 움직임을 함께 봐야 해석할 수 있다.

기본 학습·뷰어는 초기화 및 약 10초마다 명령을 무작위로 바꾼다. 방향 명령 모드에서는 다음 식으로 yaw 명령을 만든다.

$$
\omega_z^{\mathrm{cmd}}=\operatorname{clip}\left(0.5\operatorname{wrap}_{[-\pi,\pi]}(\psi^{\mathrm{target}}-\psi),-1,1\right).
$$

\(\psi\)는 현재 yaw 방향각(rad), \(\psi^{\mathrm{target}}\)은 목표 방향각, wrap은 각도 차이를 최단 회전 방향으로 정리하는 연산이다. 전진과 회전 명령이 함께 있으면 곡선 운동, 음의 전진 명령이면 후진할 수 있다. 특정 영상이 명령을 따르는지 여부는 해당 실행의 명령 기록으로 확인해야 한다.

정량평가에서는 전진 0.5 m/s, 횡속도·yaw 0으로 고정한다. 영상 궤적을 적분 명령선과 비교한 위치 차이를 정식 경로 추종 RMSE라고 부르지 않는다.

## 29페이지 · 정량 결과의 직관적 해석

**슬라이드 수식**

$$
\frac{0.0961}{0.5}\times100=19.2\%,
$$

$$
x_{\mathrm{ideal}}^{20\mathrm{s}}=0.5\ \mathrm{m/s}\times20\ \mathrm{s}=10\ \mathrm{m}.
$$

첫 식은 RMSE를 명령 속도 크기로 정규화한 것이다. “평균 실제 속도가 19.2% 부족하다” 또는 “추종 정확도 80.8%”라는 뜻이 아니다. 초과 속도와 부족 속도 모두 RMSE를 키울 수 있다.

둘째 식은 20초 내내 이상적으로 몸체 전진 속도 0.5 m/s를 유지했을 때의 적분 기준이다. 실제 보고된 진행량 9.12 m는 몸체 전진 속도 적분의 episode 평균이다. 조기 종료한 시행은 종료 전까지 누적한다. 세계좌표에서 출발점과 도착점 사이 거리가 9.12 m라는 뜻은 아니다.

**발표 설명:** “대부분은 20초를 생존했지만, 속도 오차는 명령 크기의 약 19% 수준이며 안정 회복 기준을 달성하지 못한 시행도 남아 있습니다.”

## 30페이지 · 고장 조건별 안정 회복의 한계

현재 PPT의 두 히트맵은 **Student 회복률**과 **Student−Teacher 회복률 차이**다. 생존율 히트맵이 아니다.

**보충 정의**

$$
H_R(j,d)=\frac{1}{N_{j,d}}\sum_{i\in\mathcal I_{j,d}}R_i,\qquad
\Delta H_R(j,d)=100\left(H_R^S(j,d)-H_R^T(j,d)\right).
$$

\(\mathcal I_{j,d}\)는 관절 j와 열화율 d인 시행들의 집합, \(N_{j,d}=144\)는 그 시행 수다. 3개 평가 시드×3개 고장 시점×16회 반복이다. \(R_i\)는 21페이지 회복 성공 여부다. 오른쪽 그림 단위는 %p다.

- 왼쪽 뒤 hip 완전 상실: Student 생존 92.36%, 회복 3.47%.
- 왼쪽 뒤 thigh 80% 저하: Student 생존 97.92%, 회복 26.39%.
- 오른쪽 뒤 calf 80% 저하: Student 생존 95.83%, 회복 36.81%.

**생존·회복의 관계를 설명하는 보충 식**

$$
H_{S\cap\neg R}(j,d)=\frac{1}{N_{j,d}}\sum_{i\in\mathcal I_{j,d}}S_i(1-R_i).
$$

\(S_i\)는 완주 여부다. 이 값은 “살아남았지만 회복 기준을 달성하지 못한 비율”이다. 왼쪽 뒤 hip 완전 상실에서 90.28%다. 회복 후 종료된 시행도 있으므로 단순히 생존율−회복률로 대체하면 안 된다.

**발표 설명:** “평균 생존율만 보면 보이지 않는 추종 회복 실패가 특정 관절·열화율에서 집중됩니다. 다만 같은 hip이나 calf 안에서도 다리별 차이가 커, 특정 관절 종류 전체가 취약하다고 일반화하지는 않습니다.”

수치만으로 latent 지연이나 특정 역학적 원인을 확정하지 않는다. 검은 테두리는 분석 대상으로 선택한 셀이며 통계적 유의성 표시가 아니다.

## 31페이지 · History 의존성

**보충 표현**

$$
z_t^{\mathrm{actual}}=\phi(h_t),\qquad
z_t^{\mathrm{zero}}=\phi(0),\qquad
z_{i,t}^{\mathrm{shuffled}}=\phi(h_{\sigma(i),t}).
$$

\(i\)는 병렬 로봇 번호, \(\sigma\)는 로봇 간 이력을 바꾸는 순열이다. Shuffled는 현재 설명에서 로봇 간 이력을 교환하는 개입이며, 한 이력 안의 시간 순서만 무작위로 섞는 것과 구분한다.

입력을 바꿨을 때 성능이 하락하면 해당 이력 입력에 정책이 의존한다는 근거가 된다. 하지만 이 실험만으로 latent가 고장 관절·열화율을 정확히 식별한다거나, 학습 중 보지 못한 모든 상황에 일반화한다고 주장할 수는 없다. 정상 평가 분포와 다른 입력을 준 영향도 포함한다.

## 32페이지 · 진행 현황과 한계

별도 수식은 없다. 완전 상실 회복률 45.89%는 다음 조건부 집계다.

$$
\widehat P(R=1\mid d=1)=0.4589.
$$

이는 현재 학습된 Student와 현재 평가 프로토콜의 결과다. “고장 상황 전체의 성공 확률”로 일반화하지 않는다. 세 평가 시드를 사용한 것과 독립적인 세 학습 정책을 얻은 것은 다르다.

## 33페이지 · 결론

별도 수식은 없다. 21페이지의 서로 다른 지표를 함께 해석한다.

**발표 설명:** “관측 이력만 사용하는 Student가 현재 평가 조건에서 높은 생존율을 유지했습니다. 그러나 완전 출력 상실의 안정 회복률은 45.89%로, 심한 고장에서의 추종 회복이 남은 개선 과제입니다.”

95.01% 생존, 81.66% 회복, 45.89% 완전 상실 회복은 각각 분모·조건이 다르다. 하나의 성공률로 합쳐 말하지 않는다.

## 34페이지 · Appendix A1 — PPO 설정

**슬라이드 PPO 수식**

$$
L_{\mathrm{PPO}}(\theta)=\mathbb E_t\left[
\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

이 식의 \(r_t(\theta)\)는 **환경 보상 \(r_t\)가 아니라 확률비**다. 혼동을 피하려면 설명할 때 \(\varrho_t\)라고 이름을 바꿔 읽어도 된다.

$$
\varrho_t(\theta)=\frac{\pi_\theta(a_t\mid x_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid x_t)}.
$$

| 기호 | 의미 |
|---|---|
| \(x_t\) | 현재 관측·이력·필요한 물리정보 등 정책 경로의 입력을 통칭 |
| \(\theta\) | 현재 학습할 정책 및 관련 encoder 파라미터 |
| \(\theta_{\mathrm{old}}\) | rollout을 수집한 시점의 정책 파라미터 |
| \(a_t\) | rollout에서 실제 선택한 행동 |
| \(\hat A_t\) | 그 행동이 기준 가치보다 얼마나 좋았는지 추정한 advantage |
| \(\epsilon=0.2\) | clipping 폭, 확률비 비교 범위 0.8~1.2 |

Advantage가 양수인 행동은 확률을 높이고, 음수인 행동은 낮추도록 유도한다. 다만 한번에 정책을 너무 크게 바꾸는 이득을 clip으로 제한한다. 확률비가 실제로 언제나 0.8~1.2 안에 있도록 강제한다는 뜻은 아니다.

**부호 주의:** 위 식은 최대화할 surrogate objective다. 코드와 19페이지에서는 그 음수를 \(L_{\mathrm{policy}}\)로 두고 최소화한다. 같은 L 표기를 쓰지만 최적화 방향을 구분해야 한다.

**GAE 보충 식**

$$
\delta_t=r_t+\gamma V(x_{t+1})-V(x_t),\qquad
\hat A_t=\sum_{\ell=0}^{H-1}(\gamma\lambda_{\mathrm{GAE}})^\ell\delta_{t+\ell}.
$$

여기서 \(r_t\)는 다시 **환경 보상**이다. \(V\)는 Critic 가치 예측, \(\delta_t\)는 TD 오차, \(H\)는 사용 가능한 rollout 구간 길이다. 위 식은 종료 mask를 생략한 개념식이며 실제 구현에서는 종료와 timeout bootstrapping을 처리한다.

- \(\gamma=0.99\): 미래 보상을 할인한다.
- \(\lambda_{\mathrm{GAE}}=0.95\): 여러 시간 길이의 advantage 추정을 섞어 편향·분산 균형을 조절한다.
- Desired KL=0.01: 이전·현재 정책 분포 차이를 보고 학습률을 조절하는 기준이다. KL을 항상 정확히 0.01로 만드는 제약이 아니다.
- Max grad norm=1.0: gradient 크기가 너무 커지면 norm을 제한한다.

**발표 설명:** “수집 당시 정책과 현재 정책의 행동 확률비를 이용해 좋은 행동을 강화하되, clipping으로 지나치게 큰 정책 변화를 억제합니다.”

## 35페이지 · Appendix A2 — PPO 학습 단위

**보충 계산**

$$
B=4096\times24=98{,}304,\qquad
B_{\mathrm{mini}}=\frac{98{,}304}{4}=24{,}576.
$$

| 단위 | 의미 |
|---|---|
| Transition | 환경 하나에서 행동 한 번을 적용해 얻은 다음 상태·보상·종료 정보 |
| Rollout | 여러 환경에서 일정 step 수집한 transition 묶음 |
| Minibatch | rollout 데이터 중 한 번의 gradient 계산에 사용하는 일부 |
| Update | optimizer로 가중치를 한 번 갱신 |
| Epoch | 현재 rollout 데이터 전체를 한 번 사용 |
| Iteration | 새 rollout 수집 + 그 데이터를 정해진 epoch 수만큼 학습 |

$$
1\ \text{iteration}=1\ \text{new rollout}+5\ \text{epochs}=20\ \text{updates}.
$$

동일 transition을 5번 사용한다고 서로 다른 transition 5개로 세지 않는다. transition 수와 sample 처리 횟수는 구분한다.

## 36페이지 · Appendix A3 — 71500의 학습량

**슬라이드 수식**

$$
71{,}500-43{,}000=28{,}500.
$$

이것은 번호 차이다. 해당 로그에서는 43,000 iteration의 학습도 수행한 뒤 기록하므로, 양 끝을 포함한 실제 수행 횟수는 다음과 같다.

$$
N_{\mathrm{iter}}=71{,}500-43{,}000+1=28{,}501.
$$

$$
N_{\mathrm{transition}}=28{,}501\times98{,}304=2{,}801{,}762{,}304,
$$

$$
N_{\mathrm{batch\ epoch}}=28{,}501\times5=142{,}505,
$$

$$
N_{\mathrm{update}}=28{,}501\times5\times4=570{,}020.
$$

모두 **현재 JT 구간**의 계산이며 Teacher 사전학습량은 포함하지 않는다. 142,505는 서로 다른 rollout에 대한 epoch 수의 합이지, 고정된 하나의 데이터셋을 142,505회 반복했다는 뜻이 아니다.

71,500은 transition 수도, 총 epoch 수도 아니다. checkpoint에 붙은 global iteration 번호다.

## 37페이지 · Appendix A4 — 학습 시간과 반복 횟수

**슬라이드 수식**

$$
5\ \text{epochs}\times4\ \text{minibatches}=20\ \text{updates}.
$$

5 epochs는 매번 새로 수집한 rollout 한 묶음을 5회 재사용한다는 뜻이다. 전체 학습이 5회라는 뜻이 아니다.

**기록된 시간의 환산**

$$
49{,}999.59/3600\approx13.889\ \mathrm{h}\approx13\ \mathrm{h}\ 53\ \mathrm{min},
$$

$$
49{,}741.70/3600\approx13.817\ \mathrm{h}\approx13\ \mathrm{h}\ 49\ \mathrm{min}.
$$

첫 값은 첫 완료 로그부터 71,500 완료 로그까지의 실제 경과시간, 둘째는 iteration_time 합이다. 기록 구간·오버헤드 때문에 일치하지 않을 수 있다. Teacher 사전학습·후속 평가 시간을 포함한 전체 프로젝트 소요시간은 아니다.

병렬 환경의 시뮬레이션 시간을 합한 값과 실제 벽시계 시간도 다르다. 4,096개 환경을 병렬 실행하기 때문이다.

## 38페이지 · Appendix A5 — 보상 구현과 해석

12페이지에서 생략한 원시 함수의 세부 정의를 설명한다.

**행동 변화 비용**

$$
f_{\mathrm{action},t}=\sum_{j=1}^{12}(a_{j,t}-a_{j,t-1})^2.
$$

이 식에는 \(\Delta t\)로 나누는 연산이 없다. 실제 시간 미분이 아니라 step 사이 변화량 비용이다.

**충돌 비용**

$$
f_{\mathrm{collision},t}=\sum_{b\in\mathcal B_{\mathrm{thigh,calf}}}
\mathbf1\{\|\mathbf F_{b,t}\|_2>0.1\ \mathrm N\}.
$$

\(b\)는 rigid body, \(\mathbf F_{b,t}\)는 접촉력이다. 접촉력 자체를 합하는 것이 아니라 임계값을 넘은 body 개수를 센다. 몸체 종료 조건 1 N과 구분한다.

**관절 위치 한계 비용**

$$
f_{\mathrm{limit},t}=\sum_j\left[(q_j^{\min,soft}-q_{j,t})_+
+(q_{j,t}-q_j^{\max,soft})_+\right],\qquad(x)_+=\max(x,0).
$$

soft min/max는 원래 관절 범위 중심을 유지하면서 범위 폭을 90%로 줄인 값이다. 허용 범위를 벗어난 각도량(rad)을 합하며, 제곱하는 비용이 아니다.

**발 체공 시간 보상**

$$
f_{\mathrm{air},t}=\mathbf1\{\|\mathbf c_{xy,t}\|_2>0.1\ \mathrm{m/s}\}
\sum_f(T_{f,t}^{\mathrm{air}}-0.5\ \mathrm s)I_{f,t}^{\mathrm{first\ contact}}.
$$

여기서 \(f\)는 발 번호, \(T_{f,t}^{\mathrm{air}}\)는 해당 발의 체공 시간, \(I^{\mathrm{first\ contact}}\)는 필터링된 첫 착지 여부다. 앞의 고장 시점 \(T_f\)와 문맥이 다르다. 체공시간이 0.5초보다 짧으면 이 항이 음수일 수 있다. 현재 코드의 접촉 필터는 현재·직전 접촉 여부를 함께 사용한다.

**발표 설명:** “기존 보상은 추종뿐 아니라 행동의 급변, 불필요한 관절 운동, 충돌, 관절 한계 접근을 함께 억제합니다. 생존율과 회복률은 이 보상과 별도로 평가합니다.”

## 39페이지 · Appendix A6 — Saving the Limping 결과

별도 공식 수식은 없지만 표의 환산을 설명할 수 있어야 한다.

$$
\frac{0.47}{0.5}\times100=94\%,\qquad
\frac{0.42}{0.5}\times100=84\%,
$$

$$
20\times0.565=11.30\ \mathrm s,\qquad20\times0.447=8.94\ \mathrm s.
$$

평균 전진 속도는 목표 대비 84%→94%, 평균 생존시간은 8.94→11.30초다. 관절 잠금 조건의 논문 결과이며, 우리 출력 상실 조건과 다른 고장이다.

평균 속도에서 목표를 뺀 값은 평균 편향과 관련되며, 각 step 오차를 제곱해 평균하는 RMSE와 다르다. 56.5%를 완주율로 읽지 않는다.

## 40페이지 · Appendix A7 — RMA 결과

**보충 계산**

$$
73.5-62.4=11.1\ \text{%p},\qquad76.2-73.5=2.7\ \text{%p}.
$$

RMA의 해당 과제 성공률은 Robust보다 11.1%p 높고, 물리정보를 쓰는 Expert보다 2.7%p 낮다. 정규화 평균 생존 0.85는 허용시간의 평균 85%를 생존했다는 뜻이다. 우리 20초 완주율과 정의가 같다거나, 85%가 단일 관절 완전 상실 회복률이라는 뜻이 아니다.

## 41페이지 · Appendix A8 — ADAPT 결과

별도 수식은 없다. ADAPT도 \(d=1\) 완전 출력 상실을 평가했다. Fig. 3의 열화율 축은 왼쪽부터 1, 0.9, 0.8, 0.6, 0.3이다.

**결과 해석용 보충 식**

$$
\overline G=\frac1N\sum_{i=1}^{N}\sum_{t=0}^{T_i-1}r_{i,t}.
$$

이는 누적 보상의 표본 평균이라는 개념식이다. \(N\)은 시행 수, \(T_i\)는 각 시행의 기록 길이, \(r_{i,t}\)는 ADAPT 자체의 보상이다. 논문 Fig. 3은 1,024개 병렬 시행에서 넘어지거나 1,000 step에 도달할 때까지의 누적 reward를 보고한다.

왼쪽 앞 calf 완전 상실에서 ADAPT 26.44, Teacher 26.54다. 두 정책의 누적 점수가 가깝다는 의미이지, ADAPT 회복률이 26.44%이거나 생존율 99.6%라는 뜻은 아니다. 누적 보상은 여러 항을 섞으므로 생존율·회복률로 역산할 수 없다.

## 42페이지 · Appendix A9 — 평가 지표의 집계

### 42-1. episode별 RMSE를 평균

**슬라이드 수식**

$$
e_i=\sqrt{\frac{\sum_{t=1}^{K_i}(v_{x,i,t}-v_{x,i,t}^{\mathrm{cmd}})^2}{\max(1,K_i)}},
\qquad\mathrm{RMSE}_{\mathrm{reported}}=\frac1N\sum_{i=1}^{N}e_i.
$$

| 변수 | 의미 |
|---|---|
| \(i,N\) | episode 번호, 전체 episode 수 |
| \(K_i\) | i번째 episode의 고장 후 유효 sample 수 |
| \(e_i\) | i번째 episode의 전진 속도 RMSE, m/s |
| reported RMSE | episode별 RMSE의 산술평균 |

모든 step을 합쳐 한 번에 계산하는 다음 값과 일반적으로 다르다.

$$
e_{\mathrm{pooled}}=\sqrt{\frac{\sum_i\sum_t(v_{x,i,t}-v_{x,i,t}^{\mathrm{cmd}})^2}{\sum_iK_i}}.
$$

현재 보고값은 episode별 동일 가중, pooled 값은 길이가 긴 episode의 step이 더 많이 기여한다. 고장 전 종료로 \(K_i=0\)이면 현재 코드에서는 연속 지표를 0으로 기록하므로, 작은 RMSE만 보고 좋은 정책이라고 판단하지 않는다. 생존·회복 실패와 함께 읽어야 한다.

### 42-2. 몸체 전진 진행량

$$
D_i=\sum_{t=1}^{K_i}v_{x,i,t}\Delta t.
$$

\(D_i\)는 m, \(\Delta t=0.02\)초다. 몸체 x방향 속도를 적분한 **부호 있는 전진 진행량**이다. 세계좌표의 경로 길이, 목표 궤적까지의 오차, 출발점과 도착점 사이 거리와 다르다.

### 42-3. 발 미끄럼

**보충 식**

$$
C_{f,t}=\mathbf1\{F_{z,f,t}>1\ \mathrm N\},\qquad
u_t=\frac{\sum_f C_{f,t}\|\mathbf v_{f,xy,t}\|_2}{\max(1,\sum_f C_{f,t})}.
$$

\(f\)는 발 번호, \(F_z\)는 수직 접촉력, \(\mathbf v_{f,xy,t}\)는 발의 수평 속도다. 접촉 중인 발의 수평 속도 크기를 평균한 뒤 유효 step 평균을 구한다. 단위는 m/s다. 접촉 발이 없으면 해당 step 값은 0이다.

### 42-4. 행동 크기 임계값 초과 비율

**슬라이드 수식**

$$
\operatorname{mean}_{j,t}\left(\mathbf1[|a_{j,t}|\geq0.98]\right).
$$

\(a_{j,t}\)는 관절 j의 무차원 행동이다. 임계값 0.98 이상인 비율을 관절·시간에 걸쳐 계산한다.

**중요한 표기 주의:** 슬라이드는 이를 ‘행동 한계 근처 비율’이라고 부르지만, 해당 JT 저장 설정의 `clip_actions`는 **100.0**이고 Actor는 tanh로 [−1,1]을 강제하지 않는다. 따라서 이 지표는 실제 clip 경계 근처 비율이나 모터 토크 포화율이 아니다. 정확히는 **‘행동 절댓값 0.98 이상 비율’이라는 진단 지표**로 설명해야 한다. 낮은 값이 곧 물리적 토크 포화 감소를 입증하지 않는다.

## 43페이지 · Appendix A10 — Encoder 구조와 출처

**슬라이드 수식**

$$
z_t^T=\mu(\xi_t)\in\mathbb R^8,\qquad z_t^S=\phi(h_t)\in\mathbb R^8.
$$

Teacher는 45D 물리정보, Student는 50×48 이력을 입력으로 받는다. 두 출력 모두 8D이며 차원이 같아야 원소별 혼합과 latent 거리 계산을 할 수 있다.

**Teacher MLP**

$$
45\longrightarrow512\longrightarrow256\longrightarrow128\longrightarrow8.
$$

은닉층에 ELU를 사용한다. [512,256,128]과 latent 8D는 Saving the Limping의 구조를 참고했고 입력 45D는 현재 프로젝트 구성이다.

**Student**

$$
50\times48\ \xrightarrow{\mathrm{Linear}(48,32)+\mathrm{ELU}}\ 50\times32
\ \xrightarrow{\mathrm{temporal\ CNN}}\ 3\times32
\ \xrightarrow{\mathrm{flatten}}\ 96
\ \xrightarrow{\mathrm{Linear}(96,8)}\ 8.
$$

시간 방향 CNN의 (kernel size, stride)는 (8,4), (5,1), (5,1)이다. 채널 수는 모두 32다.

**시간 길이 계산**

$$
L_{\mathrm{out}}=\left\lfloor\frac{L_{\mathrm{in}}-k_c}{s_c}\right\rfloor+1
\quad\text{(padding 0, dilation 1)},
$$

$$
50\xrightarrow{(8,4)}11\xrightarrow{(5,1)}7\xrightarrow{(5,1)}3.
$$

\(L_{\mathrm{in/out}}\)은 시간축 길이, \(k_c\)는 kernel size, \(s_c\)는 stride다. \(k_c\)는 학습 iteration k와 다르다. 마지막에는 32채널×3시점=96개 값을 펴서 8D로 선형 투영한다.

시간 CNN은 RMA에 기술된 구조를 계승한다. 현재 프레임 임베딩은 Linear 한 층+ELU이며 RMA 원문의 2층 MLP 설명과 다르다. 50프레임의 길이와 시간 길이를 구분한다. 현재 주기에서는 약 1초다.

**‘압축은 정해진 방법인가, 학습되는가?’**

개발자가 층 수·차원·CNN kernel/stride를 정하고, 각 층의 가중치는 PPO와 모방 손실로 학습한다. 입력을 되살리는 reconstruction loss가 있는 일반적인 autoencoder라고 부르지 않는다. latent 8개를 미리 지정한 물리량 8개로 해석하지 않는다.

**발표 설명:** “Teacher와 Student는 서로 다른 정보를 입력받아 동일한 8차원 특징 공간으로 변환합니다. 구조는 선행연구를 참고했지만 입력 구성은 프로젝트에 맞췄으며, 특징을 만드는 가중치는 학습으로 얻습니다.”

---

## 발표 전 확인할 표기·해석 주의사항

1. **08페이지:** rollout 식은 4,096개 환경을 곱해야 98,304 transitions가 된다.
2. **09페이지:** \(\gamma=0.99\)는 할인율, \(\rho_0\)는 초기분포다. Student는 부분 관측 문제를 푼다.
3. **10·18페이지:** 행동 scale 0.25는 degree→radian 변환이 아니며, 행동이 자동으로 [−1,1]이라는 뜻도 아니다.
4. **11페이지:** 슬라이드 고장 식은 요약 모델이다. 실제 적용에는 motor strength·PD 배율·토크 clip도 있다.
5. **13페이지:** tracking_sigma 0.25는 지수 보상 폭이며 action scale과 별개다.
6. **17·19페이지:** \(\alpha\)는 입력 특징 혼합, \(\beta\)는 모방 손실 가중치다. 네트워크 가중치를 두 모델 사이에 나눠 주는 비율이 아니다.
7. **19페이지:** 모방 loss는 제곱하지 않은 평균 L2 거리이며, detach는 모방 경로의 Teacher gradient만 차단한다.
8. **21·26페이지:** 회복 성공 후에도 종료될 수 있다. 생존·회복은 포함관계가 보장되지 않는다.
9. **29·42페이지:** 몸체 전진속도 적분은 세계좌표 경로 거리나 경로 RMSE가 아니다.
10. **30페이지:** 두 그림은 모두 회복률 관련이다. 생존과의 관계는 같은 episode의 생존/회복을 교차 집계해야 한다.
11. **34페이지:** PPO 확률비와 환경 reward가 모두 r로 표기되어 있다. 또한 표시된 PPO 식은 최대화 목적, 19페이지는 최소화 손실이다.
12. **36·37페이지:** 71,500은 global iteration 번호다. 5 epochs는 매 rollout의 재사용 횟수다.
13. **42페이지:** 0.98 이상 행동 비율을 실제 action clip/모터 토크 포화율이라고 부르지 않는다. 저장 action clip은 100.0이다.

## 근거 파일

- [v12 수식 목록](assets/v12_math_preview.json)
- [v12 기존 발표 노트](presentation_notes_v12.md)
- [관측·이력 차원 정의](../../legged_gym/envs/a1_official_wim_teacher/joint_schema.py)
- [Teacher 환경·PD 토크 구현](../../legged_gym/envs/a1_official_wim_teacher/a1_official_wim_teacher.py)
- [Joint 환경·관측·고장 시점 구현](../../legged_gym/envs/a1_official_wim_teacher/a1_official_wim_joint.py)
- [Encoder·특징 혼합·모방 손실](../../legged_gym/learning/joint_teacher_student_actor_critic.py)
- [PPO 손실과 가중치 갱신](../../legged_gym/learning/teacher_ppo.py)
- [기본 환경·보상·종료 구현](../../legged_gym/envs/base/legged_robot.py)
- [A1 보상 설정](../../legged_gym/envs/a1/a1_config.py)
- [평가 프로토콜](../../legged_gym/evaluation/p5_paired_protocol.json)
- [Onset 평가 코드](../../legged_gym/scripts/evaluate_teacher243_failure_onset.py)
- [JT 저장 설정](../../logs/jt_wim243_failure_fullrange_onset/Aug14_00-37-37_/resolved_config.json)
- [정량 평가 결과](../results/p5-tf43000-vs-jt71500.md)
- [Saving the Limping 원문](../papers/main/2023-liu-saving-the-limping/paper.md)
- [RMA 원문](../papers/main/2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md)
- [ADAPT 원문](../papers/main/2023-wu-adaptive-control-strategy-for-quadruped-robots-in-actuator-degradation-scenarios/paper.md)
- [Learning to Walk in Minutes 공식 프로젝트](https://leggedrobotics.github.io/legged_gym/)
