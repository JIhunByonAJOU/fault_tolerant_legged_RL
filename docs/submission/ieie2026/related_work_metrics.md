# 선행연구·평가 지표·비교 가능성

확인일: 2026-10-02. 원논문과 저자/연구기관 공식 공개자료를 사용한다. 코드 확인은 [public_code_audit.json](sources/public_code_audit.json)의 HEAD SHA에 한정한다. 아래 '본 연구 권장'은 선행논문을 그대로 재현했다는 뜻이 아니다.

## 핵심 선행연구

| 연구 | 대상/고장/방법 | 원문에서 확인된 평가 | 우리와 비교할 때 주의 |
|---|---|---|---|
| [Saving the Limping](https://arxiv.org/abs/2210.00474) (2023 v3) | Unitree A1, 관절 잠김, 교사-학생/공동학습 | IV-B TableI: 0.5m/s 전진 명령; 고장전후 평균전진속도; 고장후 최대20초 생존시간을20초로 정규화해 평균/P25/P50; 지형당1500인스턴스 | torque loss≠joint lock. 이 논문의 survival average는 S20 생존확률이 아님. 공개원고 All: FailureEnv 고장후0.47m/s·평균생존56.5%, BaseEnv0.42m/s·44.7%는 참고수치이며 우리와 직접순위비교 불가 |
| [RMA](https://arxiv.org/abs/2107.04034) (RSS2021) | A1, 환경/동역학변화, privileged encoder+history adaptation | V/TableII: 정규화 time-to-fall, forward reward, success, distance, adaptation samples, torque/smoothness/ground impact; 3학습정책×1000episode | 고장전용법 아님. 생존뿐 아니라 거리라는 기동성 증거를 보여주는 근거. torque/smoothness 수치를 단위/적분정의없이 재사용하지 않음 |
| [ADAPT](https://arxiv.org/abs/2312.17606) (DAI2023) | A1, 연속단일관절출력저하; 12교사→Transformer 행동복제 학생 | Fig3: 관절12×저하율{1,.9,.8,.6,.3} 누적reward heatmap, 조건당1024병렬실행; 초기/명령/고장시점무작위, 전도 또는1000step 종료 | 같은 fault 수식은 활용가능. reward 가중치·종료길이·분포가 달라 원고 reward≈27을 우리 reward와 직접비교 못함 |
| [Random Joint Masking](https://arxiv.org/abs/2403.00398) (ICRA2024) | Go1, zero torque+joint locking; masking+history joint-state estimator+curriculum | IV-A TableII: 1.0m/s 전진명령, vx RMSE, FL/RR HR/HP/KP별 결과와 component ablation. 전체RMSE0.1951, baseline0.5787; 정상0.1699 vs0.1469 | 정상성능 tradeoff도 공개. robot·평가구간·fault조건·seed수를 동일하게 만들기 전 외부성능우위 판단불가 |
| [FT-Net](https://hub.hku.hk/handle/10722/339580) (RA-L2023, DOI10.1109/LRA.2023.3329766) | 부분/완전모터고장, adaptor+모델기반지지영역 유도, 4족→3족 | 회복/지형/동작실험을 포함한다. 이번 조사에서 논문표의 metric 정의를 세부추출하지 않아 정량값 인용하지 않음 | 공식공개repo는 supplementary PDF/그림뿐. source/weights공개와 구분 |
| [AcL](https://arxiv.org/html/2503.21401v1) (2025) | Go2, 최대4관절/2다리 zero torque; 여러교사 action-style rewards→단일학생 | Fig4 style reward, Fig6 선/각속도 시계열, 고장300step/해제600step, ablation | 연속부분열화와 다른범위. 영상만으로 정량성능우위 주장불가 |
| [Mixture-of-Experts RL](https://arxiv.org/html/2606.25965v1) (2026 preprint) | Go2/IsaacLab, fault-conditioned experts, 1~2다리고장 | Fig4 8192robot cumulative reward분포, monolithicPPO 대비; actor폭128/64/32 capacity연구 | 공개code+weights있으나 Go2·expert모델·입력·IsaacLab환경이 다름. reward 직비교불가 |
| [Adaptive Gait Timing](https://arxiv.org/html/2608.07328v1) (2026 preprint) | Kyon68kg/MJX, random single-joint torque loss, proprioception history+latentalignment+gaitfrequency | IV-B: 5독립학습/95%CI, 1024agent, t5s고장+20s후구간; 선/각추종오차+생존, 관절별집계, history/latent 제거실험+cosine | 우리 구조와 상당히 겹치므로 history/latent 최초 제안 불가. terrain/gaitfrequency 포함/무게/시뮬차이 명시 |

로컬 근거: `docs/papers/main/2023-liu-saving-the-limping/paper.md` IV-B; `2021-kumar-rma-rapid-motor-adaptation-for-legged-robots/paper.md` V/TableII; `2023-wu-adaptive-control-strategy-for-quadruped-robots-in-actuator-degradation-scenarios/paper.md` Fig3; `2024-kim-random-joint-masking/source.pdf` pp4–5. 최신논문은 링크의 버전을 고정하여 인용한다. preprint를 검증된 journal publication으로 표기하지 않는다.

## 채택 지표와 직관

본 연구의 정의는 평가 스크립트와 맞춘 뒤 고정한다. 특히 `$Q_{strict}$`는 **엄격허용범위 시간비율**이지 RL의 action-value `$Q(s,a)$`가 아니다. 본문에서는 `strict tracking fraction`이라 명확히 부른다.

고장시각을 $t_f$, 평가길이를 $T=20\,\mathrm{s}$, 생존종료를 $\tau\leq T$, 전도 전 조건부관측값을 사용한다.

1. **안전성**: $S_{20}=N^{-1}\sum_i\mathbf1[\tau_i\ge20\,\mathrm{s}]$; alive-time $\bar\tau=N^{-1}\sum_i\min(\tau_i,T)$ (s). 넘어짐과 지속기동을 구분한다. STL 평균정규화생존 및 RMA TTF와 개념상 연결되지만 같은수치가 아니다.
2. **기동성**: 세계좌표 명령방향 단위벡터 $\hat u_{cmd}$에 대해 $D_\parallel=\int_{t_f}^{t_f+T}v_{world}(t)^\top\hat u_{cmd}(t)dt$ (m); 전도이후기여는0. 일정0.5m/s이면 목표거리10m. $P_{cmd}=D_\parallel/(\|v_{cmd}\|T)$ (무차원), 0.7이면 목표전진거리70%. 직진에서 body vx 적분과 world진행거리의 차이를 yaw drift로 설명한다. 가변/영명령에서는 denominator를 명령거리적분으로 정의하고0을exclude한다.
3. **정지/후진**: stall-time은 생존시간 중 명령이 충분히 큰데 명령방향속도가 정한 양의 임계값이하인 시간(s); backward-time은 명령방향속도가 음의 임계값보다작은 시간(s). threshold·관찰분모·전도후처리를 명시한다. 이 두 지표는 **본 연구 선택**이며 STL/RMA의 이름을 빌려 원문지표라 쓰지 않는다.
4. **연속추종**: $\mathrm{RMSE}_x=\sqrt{\frac1{n_{alive}}\sum_{t\in alive}(v_x(t)-v_x^{cmd}(t))^2}$ (m/s), y축같음; yaw-rate RMSE는 rad/s. 조기전도 때문에 표본이 줄어든 정책이 좋아보일 수 있어 S20/alive-time과 함께 보고하고 분모를 고정한다. 'yaw RMSE'가 각도(rad)인지 yaw-rate(rad/s)인지 혼동금지.
5. **정밀성공**: 전체postfault시간분모로 $Q_{strict}=T^{-1}\int\mathbf1[alive\land |e_x|<\epsilon_x\land |e_y|<\epsilon_y\land |e_\omega|<\epsilon_\omega]dt$ (비율). 허용오차는 결과확인전에 고정, 임계값감도는 appendix. 느리지만전진하는 정책은 D/P에서 인정되면서 Q만낮을수있다.

**주결과**는 S20·전진거리/명령달성률·stall/backward·vx/vy/yaw-rate RMSE·Qstrict를 함께 표기한다. 생존만높고 D≈0이면 서있음; D>0이나P<1이면 속도를줄여기동유지; 낮은RMSE인데S20낮으면 earlyfall selection가능성이다. latent MSE/cosine은 메커니즘보조지표로 쓰고 보행성능대체지표로 쓰지 않는다.

## 비교 설계와 표현 범위

### 같은 환경에서의 주요 결과표

- 정상보행 PPO: 고장 학습이 없는 정책이 동일한 torque-loss 환경에서 얼마나 성능이 저하되는가.
- 독립적으로 학습한 원 교사 정책이 확보되고 동일한 평가 프로토콜을 적용한 경우에만 원 교사 baseline으로 비교한다. 최종 공동학습 actor에 교사 latent를 교체 주입한 결과는 **representation intervention diagnostic**이다. 학생 전용 latent와 함께 actor가 최적화된 이후 교사 latent를 넣으면 호환성이 달라질 수 있으므로 정보 상한·oracle·원 교사 baseline으로 부르지 않는다. 이 개입의 결과를 특권정보를 알면 달성 가능한 최대 성능으로 해석하지 않는다.
- 배포학생 JT77500: 본 연구의 최종 결과. 이름은 `Proposed history-conditioned policy` 등 방법의 역할로 쓰고 내부 체크포인트는 provenance에만 적음.
- history 제거·shuffle/지연·latentalignment 제거는 기여 검증용. **추론 시 history를 zero로 만든 실험**은 제거한 채 학습한 모델과 다르므로 stress test라표기한다.
- 현재 없는 방법 또는 추론 구조만 구현한 외부 방법을 원논문 재현 baseline이라고 표기하지 않음.

선행연구와 연결되는 추가 검증안으로 정상포함12관절×고장율0/.2/.4/.6/.8/1.0, 고정0.5m/s 직진+1.0m/s 민감도+회전명령, 고장 전 정상 구간5s+고장 후20s를 제안했다. **이 제안은 현재 수행한 프로토콜 설명이 아니다.** 현재 기존 평가는 고장시점2–10s 무작위 조건을 사용하므로 결과표에는 실제 onset 분포와 GPU 평가 조건을 paper/results manifest에 고정한다. 앞으로5s고정 평가를 수행한 경우에만 해당 조건으로 표기한다. seed는 모델 간 paired 초기 상태·fault time·domain randomization까지 맞춰야 한다. 병렬 환경 개수 변경은 난수 순서를 바꿀 수 있으므로 seed가 같다는 이유만으로 paired 평가를 주장하지 않는다.

episode bootstrap95%CI는 **이 체크포인트의 rollout 분산**만 반영한다. 독립 학습 seed가 없다면 학습 재현성 분산이나 알고리즘의 통계적 우위로 해석하지 않는다. checkpoint 선택에 사용한 조건과 최종 test 조건이 같으면 held-out 결과라고 부르지 않는다.

### 외부논문 표

외부 논문은 robot·fault·명령·지형·기간·평가 정의·실물 여부를 나란히 비교하는 **연구범위표**로제시한다. 원문 숫자와 우리 숫자를 한 순위표에 섞지 않는다. 'STL과 같은 0.5 m/s 명령을 사용했다'는 가능하지만, 'STL 대비 X% 향상'은 joint lock ↔ torque loss와 지형·종료 분포 차이를 없애기 전에는 불가하다.

## 공개 코드/가중치 실행 가능성

| 공식 자료 | 확인 SHA/공개상태 | 실행 장벽 |
|---|---|---|
| [RMA code](https://github.com/antonilo/rl_locomotion), [공식project](https://ashish-kmr.github.io/rma-legged-robots/) | `f71db819` base `full_22000.pt`, `policy_22000.pt` 제공. 현재 README는 Cross-Modal Supervision의 RMA 변형이라고 명시 | 원 RMA 배포 학생 가중치인지 확인되지 않음. 제공 base weights는 CMS 훈련 guide로 설명된다. RaiSim native 환경·입력·정규화·행동을 확인해야 하며 이를 원 RMA 전체라 쓰지 않음 |
| [ADAPT](https://github.com/WentDong/Adapt) | `5db91c73`, teacher/student source 있음, tree71개에 pt/pth/onnx/ckpt 없음 | 가중치 미공개. dataset·teacher 12개·student 훈련 필요; 현재 inference로 외부 최종 model 비교 불가 |
| [AcL](https://github.com/ACT-legmotion/AcL-ActionLearner) | `1ccf04b3`, 영상·그림·README 24파일, weights없음 | source 공개 약속과 현재 공개 실체 구분 |
| [FT-Net](https://github.com/arclab-hku/FT-Net_Quadruped_Robots) | `a39d438b`, supplementary PDF·figure·README 3파일 | 학습 source·weights 없음 |
| [IIT fault locomotion](https://github.com/iit-DLSLab/fault-locomotion-isaaclab) | `d7892255`, Go2 PPO `model_19950.pt`/`19999.pt`, exportedpolicy.pt/onnx | IsaacLab+Go2+customrsl_rl환경; 공개 PPO가 MoE인지 plain PPO인지 모델 key 검증 필요. A1 Isaac Gym에 direct load 금지 |
| [Adaptive Gait Timing project](https://gianni0907.github.io/fault_tolerant_locomotion/) | paper·video 확인, 코드·weight 링크 미확인 | Kyon MJX환경. 현시점 reproduction 실행 가능이라 주장 불가 |
| [Random Joint Masking project](https://sites.google.com/view/learning-impaired-joints-loco) | paper·videos만 확인; 공식 code·weights 링크 없음 | 없는 weight를 임의 repo로 대체하지 않음 |

이 파일 작성에서는 학습·외부checkpoint실행을 하지 않았다. 공개 가중치를 찾았다는 사실과 공정 비교를 실행했다는 사실은 별개다.

## 문제 정의·기여 제안

'교사-학생·history·PPO로 fault를 해결한 최초 연구'는 위 선행연구 때문에 성립하지 않는다. 현 자료로 설득 가능한 기여 후보는 **같은 환경에서 최종 학생의 fault 후 기동성 보존을 입증하고, 생존과 엄격 추종의 차이를 관절별 연속 열화 분석으로 설명**하는 것이다. 다만 지표 추가 자체를 새 제어 알고리즘 기여라고 부르지 않는다. 저학습률·폭·handoff가 우리 방법의 실질 수정이면 대응 ablation 및 훈련 비용과 함께 검증해야 한다. 성능 좋은 조건만 선택한 최고 주장은 금지하고 worst-joint 및 complete-loss 실패 조건을 함께 공개한다.
