# Fault-Tolerant Quadruped Locomotion

Isaac Gym에서 Unitree A1의 **단일 관절 모터 출력이 갑자기 저하되어도 계속 보행하는 정책**을 학습하고 평가하는 연구 프로젝트입니다. 고장 정보를 직접 사용하는 privileged Teacher와, 실제 배포 시 사용할 수 있도록 관측 history만 사용하는 Student를 공동 학습합니다.

<p align="center">
  <img src="docs/assets/jt71500_fault_adaptation.gif" width="840" alt="JT71500 student policy walking with a degraded joint in Isaac Gym">
</p>

<p align="center"><em>JT71500 Student-only policy in Isaac Gym. 색상으로 표시된 관절이 출력 저하 대상입니다. 이 영상은 정성적 시연이며, 성능 수치는 아래의 전체 조건 평가에서 산출했습니다.</em></p>

## 연구 목적

일반적인 보행 정책은 모든 액추에이터가 정상이라는 가정 아래 학습됩니다. 하지만 한 관절의 모터가 약해지면 동일한 관절 명령을 보내도 실제 토크가 충분히 발생하지 않으며, 자세와 속도 오차가 누적되어 넘어질 수 있습니다.

이 프로젝트는 다음 질문을 다룹니다.

> 고장 관절과 저하율을 직접 알려주지 않아도, 로봇이 최근 관측의 변화만으로 고장을 추론하고 보행을 유지할 수 있는가?

현재 범위는 **Isaac Gym 시뮬레이션 검증**입니다. 실물 로봇 성능이나 sim-to-real 전이를 주장하지 않습니다.

## 고장 모델

12개 관절 중 하나를 선택해 명령 토크를 다음과 같이 감소시킵니다.

$$
\tau^{\mathrm{applied}}_i=(1-d_i)\tau^{\mathrm{command}}_i,
\qquad d_i\in\{0,0.2,0.4,0.6,0.8,1.0\}.
$$

- $d_i=0$: 정상 관절
- $d_i=1$: 해당 모터의 명령 토크가 완전히 사라짐
- 고장은 episode 시작 후 2–10초 사이에 발생
- 한 episode에서는 하나의 관절만 저하

이 모델은 **출력 저하(torque attenuation)**이며 관절 각도를 기계적으로 고정하는 joint locking과는 다릅니다.

## Teacher–Student 정책

```text
Privileged state 45D ── Teacher MLP ── z_T 8D ─┐
                                                ├─ fused latent ─┐
50 × history 48D ── Student 1-D CNN ─ z_S 8D ─┘                ├─ Actor ─ 12 joint actions
Current observation 235D ───────────────────────────────────────┘
```

| 구성 | 입력 | 역할 |
| --- | --- | --- |
| Teacher encoder | 고장 상태를 포함한 privileged state 45D | 학습 중 목표 latent 제공 |
| Student encoder | 최근 50 frame × 48D history | 고장 정보를 직접 받지 않고 동역학 변화를 추정 |
| Actor | 현재 관측 235D + latent 8D | 12개 관절의 position target 출력 |

공동 학습 중 latent는 다음과 같이 Teacher에서 Student로 전환됩니다.

$$
z_t=(1-\alpha)z_t^T+\alpha z_t^S,
\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{PPO}}+\beta\lVert z_t^S-\operatorname{sg}(z_t^T)\rVert_2.
$$

10,000 iteration 동안 $\alpha:0\rightarrow1$, $\beta:1\rightarrow0$으로 변화합니다. 최종 Student는 privileged failure state 없이 현재 관측과 history만으로 행동합니다.

## 검증된 JT71500 결과

선택된 배포 정책은 `model_71500.pt`의 Student-only branch입니다. Teacher와 Student를 같은 명령, 관절, 저하율, seed 및 horizon 조건으로 평가했습니다.

| 지표 | TF43000 Teacher | JT71500 Student |
| --- | ---: | ---: |
| 고정 고장 생존율 | 95.37% | **95.63%** |
| 완전 출력 상실($d=1$) 생존율 | 88.37% | **90.10%** |
| 고정 고장 전진속도 RMSE | **0.0866 m/s** | 0.1031 m/s |
| 무작위 onset 이후 생존율 | 94.20% | **95.01%** |
| onset 이후 안정 회복률 | **90.09%** | 81.66% |
| onset 이후 전진속도 RMSE | **0.0770 m/s** | 0.0961 m/s |

- 고정 고장: 정책별 3,456 episode
- 무작위 onset: 정책별 8,640 episode, 고장 후 20초 평가
- 안정 회복: 속도 오차 기준을 1초 연속 만족한 episode 비율

Student는 고장 정보를 받지 않고도 Teacher 수준의 생존율을 유지했습니다. 반면, 고장 발생 후 빠르게 목표 속도로 복귀하는 능력과 속도·yaw 추종 오차는 Teacher보다 부족합니다. 따라서 현재 결과의 의미는 **“넘어지지 않고 움직이는 능력의 전달”**에 가깝고, 완전한 Teacher 성능 복제는 아닙니다.

History intervention에서도 실제 history 사용 시 생존율은 95.60%였지만, zero history에서는 79.98%, 다른 로봇의 history를 섞으면 3.24%로 감소했습니다. 이는 Student가 현재 관측만 무시한 채 걷는 것이 아니라 시간 정보를 실제로 사용한다는 근거입니다.

상세 프로토콜과 해석은 다음 문서에 있습니다.

- [Teacher–Student 동일 조건 평가](docs/results/p5-tf43000-vs-jt71500.md)
- [공동학습 및 checkpoint 계약](docs/plan/joint-teacher-student-training.md)
- [중간 checkpoint 전환 분석](docs/results/jt-transition-audit-20260928.md)
- [onset 평가 프로토콜 진단](docs/results/onset-v2-diagnostic-20260928.md)

## Repository 구조

```text
legged_gym/
├── envs/a1_official_wim_teacher/   # Teacher, failure, joint-training environments
├── learning/                       # Teacher/Student actor-critic and PPO extensions
├── evaluation/                     # Frozen evaluation protocols and viewer components
├── scripts/                        # Training, evaluation, diagnostics, and plotting
└── tests/                          # Tensor contracts and protocol regression tests
docs/
├── plan/                           # Training and evaluation specifications
├── results/                        # Audited quantitative results
└── assets/                         # README media
```

핵심 task 이름은 다음과 같습니다.

| Task | 설명 |
| --- | --- |
| `a1_official_wim_teacher243_failure_fullrange` | privileged failure Teacher |
| `a1_official_wim_jt_failure_fullrange_onset` | random-onset joint Teacher–Student training |

## 설치

이 저장소는 [legged_gym](https://github.com/leggedrobotics/legged_gym)을 기반으로 하며 NVIDIA Isaac Gym Preview 4, Python 3.8, PyTorch와 `rsl_rl`이 필요합니다.

```bash
# Isaac Gym 설치 후
cd isaacgym/python
pip install -e .

# rsl_rl v1.0.2 설치 후
cd /path/to/rsl_rl
pip install -e .

# 본 프로젝트 설치
cd /path/to/fault_tolerant_legged_RL
pip install -e .
```

학습 checkpoint와 대용량 원시 로그는 Git 저장소에 포함하지 않습니다. 평가 스크립트의 기본 경로를 사용하려면 문서에 기록된 checkpoint를 동일한 `logs/` 구조에 배치해야 합니다.

## 현재 한계와 진행 방향

- 시뮬레이션 결과만 검증했으며 실제 A1 로봇에서는 평가하지 않았습니다.
- 기존 JT71500은 하나의 학습 seed에서 선택됐습니다. 평가 seed가 여러 개인 것과 독립 학습 반복은 다릅니다.
- 심한 고장의 일부 관절은 생존하더라도 안정 회복률과 추종 성능이 낮습니다.
- 현재는 초기 Teacher→Student 전환의 정책 변화, Student encoder 용량, 강한 고장 데이터 노출량을 분리해 재학습하고 있습니다.

## References and acknowledgements

이 프로젝트는 아래 연구와 오픈소스 구현을 기반으로 합니다. 본 저장소의 결과는 원 논문의 공식 재현 결과가 아니며, **Saving the Limping의 공동학습 구조와 RMA 계열 history adaptation을 단일 모터 출력 저하 환경에 결합한 시뮬레이션 연구**입니다.

1. D. Liu, T. Zhang, J. Yin, and S. See, **“Saving the Limping: Fault-tolerant Quadruped Locomotion via Reinforcement Learning,”** 2023. [Paper](https://arxiv.org/abs/2210.00474)
2. A. Kumar, Z. Fu, D. Pathak, and J. Malik, **“RMA: Rapid Motor Adaptation for Legged Robots,”** RSS 2021. [Paper](https://arxiv.org/abs/2107.04034)
3. X. Wu, W. Dong, H. Lai, Y. Yu, and Y. Wen, **“Adaptive Control Strategy for Quadruped Robots in Actuator Degradation Scenarios,”** DAI 2023. [Paper](https://doi.org/10.1145/3627676.3627686) · [Official code](https://github.com/WentDong/Adapt)
4. N. Rudin, D. Hoeller, P. Reist, and M. Hutter, **“Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning,”** CoRL 2021. [Paper](https://arxiv.org/abs/2109.11978) · [legged_gym](https://github.com/leggedrobotics/legged_gym)
5. V. Makoviychuk et al., **“Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning,”** 2021. [Paper](https://arxiv.org/abs/2108.10470)

원본 `legged_gym` 코드와 NVIDIA 구성요소의 저작권 및 라이선스는 [LICENSE](LICENSE)와 [`licenses/`](licenses/)를 따릅니다.
