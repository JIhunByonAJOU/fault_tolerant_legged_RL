# JT71500 Student 고장 onset / history 진단 (2026-09-21)

## 범위와 프로토콜

- **재학습 없음.** `model_71500.pt` (SHA-256 `1887f9b5abeea72ca87e0c92172687462714793f91c9011490a170cffe42e1f0`)를 Student-only 추론으로 평가.
- GPU PhysX, 전진 명령 0.5 m/s, 고장 발생 2/5/10초, 이후 20초, 12관절 × degradation `d={0.2,0.4,0.6,0.8,1.0}` × 4반복 × seed 1/2/3. 각 history 모드별 2,160 로봇, 각 severity별 432 로봇.
- `actual`: 본인 history. `zero`: 고장 시점 이후 history 50×48만 0으로. `shuffled`: 고장 시점 이후 **같은 severity·onset, 다른 관절(+6 위치)** 로봇의 history로 교체. 현재 235D 관측은 세 모드 모두 유지.
- 회복: 고장 후 `|vx−0.5|≤0.1 m/s` 및 `|yaw rate|≤0.2 rad/s`를 연속 1초 만족. 생존: 고장 후 20초 내 종료되지 않음. 두 지표는 별개.

## 결과

| d | actual 생존/회복 | zero 생존/회복 | shuffled 생존/회복 | actual vx/yaw RMSE |
|---:|---:|---:|---:|---:|
| 0.2 | 97.0% / 97.5% | 90.3% / 0.0% | 3.5% / 3.7% | 0.075 m/s / 0.082 rad/s |
| 0.4 | 96.1% / 95.1% | 88.9% / 0.0% | 3.9% / 3.0% | 0.076 / 0.090 |
| 0.6 | 97.2% / 89.4% | 87.5% / 0.0% | 4.4% / 2.5% | 0.079 / 0.108 |
| 0.8 | 94.2% / 81.0% | 71.1% / 0.0% | 3.2% / 1.4% | 0.101 / 0.143 |
| 1.0 | 90.0% / 44.7% | 43.1% / 0.0% | 2.5% / 0.5% | 0.157 / 0.247 |

전체 평균 actual 생존/회복은 94.91%/81.53%; zero 76.16%/0%; shuffled 3.52%/2.22%. **살아남는 것과 목표 속도·yaw로 안정화되는 것은 다르다.** 특히 완전 고장 `d=1.0`에서 actual은 생존 90.0%지만 회복은 44.7%뿐이다.

관절별로 보면 (각 n=36), `RL_hip d=1.0`은 생존 88.9%에도 안정 회복 **0%**다. `RR_calf d=0.8`은 생존 100%인데 회복 38.9%, `RL_thigh d=0.8`은 생존 94.4%/회복 41.7%다. 이들이 후속 진단의 우선 관절이다.

## 5초 onset 시간 추적

선택 사례 3관절 × 2반복을 10초 post horizon으로 추적했다. 한 시각의 `latent8`, 12D action, 로봇 `vx`, yaw rate를 고장 전/후로 기록했다. `action`과 latent는 그 시각 `env.step` **직전** 정책 입력/출력이고, 속도·yaw도 그 직전 상태다. 따라서 고장 설정 직후 최초 action에는 관측 history가 아직 고장 반응을 담지 못할 수 있다.

- RL hip `d=1.0` 첫 반복: 고장 전 2초 `vx RMSE=0.043 m/s`, 평균 `|yaw|=0.022 rad/s`; 고장 후 첫 2초 `0.100 m/s`, `0.155 rad/s`. pre-history latent 중앙값과의 평균 8D 거리는 `3.37→17.51`로 증가. 두 번째 반복은 10초 post horizon 도중 종료했다.
- RL thigh `d=0.8` 첫 반복: 고장 후 첫 2초 latent 거리 `4.61→8.74`, 평균 `|yaw|=0.040→0.125 rad/s`; 속도 RMSE는 첫 2초에는 낮지만 이후 2–10초 `0.145 m/s`.
- RR calf `d=0.8` 첫 반복: latent 거리 `4.12→4.14`로 뚜렷한 증가는 없고, 고장 후 속도 RMSE `0.102 m/s`. 사례 간 반응이 다르므로 latent 변화량만으로 관절 식별을 주장하지 않는다.

그림: [severity별 4개 KPI](../../logs/evaluations/jt71500_onset_history_v1/figures/severity_history.png), [시간 추적 반복 0](../../logs/evaluations/jt71500_onset_history_v1/figures/71500_onset_trace_rep0.png), [시간 추적 반복 1](../../logs/evaluations/jt71500_onset_history_v1/figures/71500_onset_trace_rep1.png). 원자료: [severity 요약](../../logs/evaluations/jt71500_onset_history_v1/severity_history_summary.json), `matrix_{actual,zero,shuffled}_seed{1,2,3}.jsonl`, `trace_actual_seed1.jsonl`.

## 판정과 한계

이번 실험은 **Student가 실제 history에 강하게 의존해 고장 후 주행한다**는 근거다. 그러나 8D가 어느 관절이 얼마나 망가졌는지 정확히 표현한다는 직접 증거는 아니다. `zero`와 다른 로봇 history는 학습 분포 밖 입력이므로 성능 붕괴 일부는 OOD 효과일 수 있다. `shuffled`는 동일 severity/onset, 다른 관절로 맞췄지만 donor 로봇의 운동 위상과 상태도 다르다.

동일 seed·지형 배정 2,160쌍을 사용했지만 GPU PhysX 재실행은 완전 결정적이지 않았다. 지형 배정 불일치 0건, 고장 **전** 종료 여부 불일치 11건/4,320개 모드 비교가 있었다. 선택 trace에서도 일부 반복의 고장 전 상태가 완전히 일치하지 않았다. 따라서 작은 차이를 개별 쌍의 엄밀한 인과효과로 해석하면 안 된다. 반면 강도 전반에서 관찰된 큰 집계 차이는 history 의존성을 지지한다.

다음 검증은 정상 운행 history와 고장 직후 history를 시간 지연/부분 마스킹하는 **보다 on-manifold에 가까운 개입**, 그리고 동일 지형·속도·운동 위상 조건에서 8D로 관절/강도를 얼마나 예측할 수 있는지 보는 진단 분류기 또는 CKA/거리 분석이다. 그 전까지 “정확한 고장 식별”은 보류한다.
