# IEIE2026 고정 과제군 2:1:1:2 — 전체 평가 원자료

**2,196 정책·시드 런 / 26,352 로봇 시행 / 13,176 Normal–Student 대응 쌍.**
완료된 첫 수집(시드1101–1103)의 고정 조건 부분집합이다. 별도 시뮬레이션이나 새로운 독립 확인 평가를 수행한 데이터가 아니다. 전진/대각/횡이동/사인파를2:1:1:2로 맞췄고, 원래12개 지형 열을 유지했다. 성능 결과에 따른 시행 선택은 없다.

## 상세 설명과 논문 작성

- [전체 데이터 상세·지형·시드·제어기·지표·필드 사전](DATA_DETAILS.md)
- [조교 피드백과 적용 범위](TA_GUIDANCE.md)
- [그림 글꼴·크기·셀 가독성·생성 방법](FIGURE_GUIDELINES.md)
- [결과 절·초록·결론의 원고 반영 기록](PAPER_RESULTS_PLAN.md)
- [Times New Roman·9 pt로 만든 논문 그림](figures/): 개별 곡선 폭 80 mm·히트맵 폭 80 mm의 PDF와 PNG, 글꼴·가독성 검증 JSON
- [당시 높이맵 hash와 일치하는 세 지형 배열](terrain/), [설정·런타임 근거](reproducibility/)

기존 `analysis/`와 `explanatory/`의 그림은 이전 진단 자료다. 논문용 새 그림은 공통 생성기 `paper_figures.py`로 만들며, `analyze.py`의 그림 생성도 같은 기준을 사용한다. 새 결과는 [복구 원고(프로시딩 7쪽·익명 6쪽, 목표 분량 미충족)](../../docs/submission/ieie2026/RESTORATION_20261017.md)에 반영했다.

## Windows에서 받기

저장소 루트에서 PowerShell 또는 Git Bash로 실행한다.

```powershell
git lfs install
git switch dev
git pull --ff-only
git lfs pull --include="datasets/ieie2026_fixed2112/raw/**"
cd datasets/ieie2026_fixed2112
py -m pip install -r requirements.txt
py analyze.py
```

`git lfs` 명령이 없다면 [Git LFS](https://git-lfs.com/)를 설치한다. CSV/JSON/그림/TeX는 일반 Git 파일이고, 전체 원시 NPZ는 같은 저장소의 Git LFS에 저장된다. 다운로드는 수GB 규모다. CSV요약이나 선택 사례만 제공하는 패키지가 아니다. 선택된26,352개 시행의 모든50Hz 유효 기록을 포함한다. 초기에 넘어지거나 정체한 시행도 빠짐없이 포함했다.

## 파일 구성

- `records.json`: 최종26,352개 시행의 전체 결과. `trial_id`, `paired_trial_id`, `source_job`, `env_id`로 원시 기록과 연결한다. `source_folder`는 이 폴더 기준 상대경로다.
- `raw/<source_job>/trace_*.npz`: 선택된 모든 로봇의 모든50Hz 표본. 시간 다운샘플링·값의 반올림·양자화 없음.
- `raw/<source_job>/trace_manifest.json`: 각 NPZ의SHA-256, 압축 전 배열payload hash, 표본 수, 필드 순서.
- `raw/<source_job>/initial_applied_state.npz`, `fault_applied.npz`: 실제 초기/물리 상태 및 고장 적용값. 원래env 인덱스를 유지하므로 `selected_env_ids.json`으로 필요한 행을 고른다.
- `raw/<source_job>/source_*.json`: 원래 수집의 완료/결과/trace manifest를 보존한 출처 증거. 원래8주행의 추가 시행도 포함하는 감사용 파일이며, 최종 구성원은 `records.json`이다.
- `analysis/`: 전체 시행CSV, 조건별 요약, 대응 차이, 원자료 색인.
- `reports/`: 기존 논문과의 비교, 손실률별 수치와 원래 그림. 한국어 용어는 토크 저하·토크 저하율을 사용한다.
- `explanatory/`: 관절별5지표, 비낙상 정체 분석, 원시 기록으로 검증한 주행 예시.
- `generated/`: `analyze.py`가 새로 만드는 Student 절대 낙상률 히트맵·5개 개별 곡선·지형 그림·정체 요약 CSV. Git에서 제외된다.
- `source_snapshot/`: 수집 당시의실행 소스와 원래프로토콜. Isaac Gym 재실행용 출처이며 Windows에서 데이터만 분석할 때는 필요 없다.
- `fixed_design.json`, `derivation_manifest.json`, `manifest.json`: 구성·출처·전체 파일 무결성. 원래Ubuntu의절대경로는 출처 기록이며 데이터 읽기에는 사용하지 않는다.

그림은 설치된 Times New Roman을 사용한다. 폰트를 찾지 못하면 자동 대체하지 않는다. 직접 지정하려면 `py analyze.py --font "C:\Windows\Fonts\times.ttf"` 또는 `py paper_figures.py --font "C:\Windows\Fonts\times.ttf"`를 사용한다. 글꼴 파일 자체는 저장소에 포함하지 않는다.

## 모든 원시 기록 검증

```powershell
py analyze.py --verify-raw --verify-only
```

모든 파일과압축 전 배열을 검증하고,각로봇의 종료시각에 맞는50Hz 행수 및고장후 표본수를 검사한다. `manifest.json`의 `raw_rows`가 정확한 전체 행수다. LFS를 받지 않고 포인터만 있는 경우 파일크기/hash 검사가 실패한다.

## 한 시행의 원본을 CSV로 보기

```powershell
py analyze.py --trial exploration_s1101_S20_r0_slot10_j01_d100_Student
```

이 예시는 생존했지만 진행이 정체된 학생 시행이다. `generated/`에모든50Hz 표본이CSV로 저장된다. 다른시행ID는 `records.json` 또는 `analysis/all_trials.csv`에서 고른다. Isaac Gym·CUDA·PyTorch는 필요 없다.

## raw NPZ 형식

배열은 `values`(N×16,float32), `flags`(uint8), `env_id`(uint16), `step`(int32)이다. step간격0.02초. flags의 비트값 1은 alive, 2는 고장 후, 4는 정체 flag다. 비트 위치는 각각 0, 1, 2다. 종료 상태는각시행의`termination`에 있다. 초기2초도보존하고,낙상/완주/상한뒤표본은생성하지 않는다.

`values`의열: world_x,world_y,body_yaw,body_vx,body_vy,body_wz,cmd_vx,cmd_vy,cmd_wz,ref_x,ref_y,path_s,furthest_s,cross_track,world_vx,world_vy. 위치·기준경로는세계좌표,속도명령과vx/vy/wz는몸체좌표다.

## 본문 그림의 역할

- **Normal–Student 비교:** 생존율·vx·vy·yaw-rate·횡오차의 다섯 개별 곡선. 각 파일에 범례·축·단위가 있다.
- **제안 정책의 성능 분포:** Student 절대 낙상률 히트맵. 관절 12개 × 저하율 5개, 셀당 216시행, 0–100% 고정 색 범위다. “어느 관절의 어느 저하에서 낙상률이 얼마인가?”를 보여준다.
- **정체 해석:** 생존율과 추종 오차를 함께 읽고, RL calf 사례를 보조 표·본문으로 설명한다.

지형 12열과 2:1:1:2는 평가 설정이다. 본문 결과의 중심은 정책 성능이며, 지형·비율 변경 효과를 주장하지 않는다. [현재 사용할 그림 목록](figures/README.md)에서 개별 PDF를 고른다. 기존 `paper_joint_vxy_gain`과 `paper_metrics_by_degradation`은 이전 시안이며 현재 본문 추천이 아니다.

## 버티기 해석

`stall_seconds>0`은30초동안새경로진행0.5m미만인구간이한번이상관찰됐다는뜻이다. 완전한정지나영구실패판정과같지않고,정체후완주한경우도포함한다.

`생존 시행 중 정체 경험 비율 = (생존했고 정체flag가 있었던 시행 수) / (생존 시행 수)`

은높은생존율을해석하는보조통계다. RL calf100%저하에서는Normal185/188=98.40%,Student4/198=2.02%다. 미완주상한까지남은정체로한정하면각각185/188=98.40%,3/198=1.52%다. 분모를전체시행216회와혼동하지않는다. 이것을새정책의주보상이나유일한성능점수로사용하지않는다.

생존율은낙상0/비낙상1의평균이다. 정체·저속으로조기종료하지않았다. 오차는고장후낙상/완주/상한전유효구간의시행별RMSE평균이다. 유효시간과생존율을함께읽는다. 평가시드3개는독립학습3회가아니다.

2026-10-17: 현재 히트맵은 낙상률(100−생존율), 낮은 값은 밝고 높은 값은 진하다. `paper_student_survival_heatmap.*`은 이전 생존율 시안으로 보존하며 현재 원고에서는 사용하지 않는다. 데이터·생존율 정의는 바꾸지 않았다.
