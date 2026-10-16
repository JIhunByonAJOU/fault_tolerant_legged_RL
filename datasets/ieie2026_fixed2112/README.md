# IEIE2026 고정 과제군 2:1:1:2 — 전체 평가 원자료

**2,196 정책·시드 런 / 26,352 로봇 시행 / 13,176 Normal–Student 대응 쌍.**
완료된 첫 수집(시드1101–1103)의 고정 조건 부분집합이다. 별도 시뮬레이션이나 새로운 독립 확인 평가를 수행한 데이터가 아니다. 전진/대각/횡이동/사인파를2:1:1:2로 맞췄고, 원래12개 지형 열을 유지했다. 성능 결과에 따른 시행 선택은 없다.

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
- `generated/`: `analyze.py`가 새로 만드는 평면속도RMSE 히트맵·정체 진단·CSV. Git에서 제외된다.
- `source_snapshot/`: 수집 당시의실행 소스와 원래프로토콜. Isaac Gym 재실행용 출처이며 Windows에서 데이터만 분석할 때는 필요 없다.
- `fixed_design.json`, `derivation_manifest.json`, `manifest.json`: 구성·출처·전체 파일 무결성. 원래Ubuntu의절대경로는 출처 기록이며 데이터 읽기에는 사용하지 않는다.

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

배열은 `values`(N×16,float32), `flags`(uint8), `env_id`(uint16), `step`(int32)이다. step간격0.02초. flags의bit1(값1)은alive,bit2(값2)는고장후,bit3(값4)는정체flag다. 종료 상태는각시행의`termination`에 있다. 초기2초도보존하고,낙상/완주/상한뒤표본은생성하지 않는다.

`values`의열: world_x,world_y,body_yaw,body_vx,body_vy,body_wz,cmd_vx,cmd_vy,cmd_wz,ref_x,ref_y,path_s,furthest_s,cross_track,world_vx,world_vy. 위치·기준경로는세계좌표,속도명령과vx/vy/wz는몸체좌표다.

## 본문 히트맵 한 개를 고른다면

**평면속도 추종RMSE의개선폭(Normal−Student)**을추천한다. 각시행의정의는

`RMSE_vxy = sqrt(RMSE_vx**2 + RMSE_vy**2)`

이며 시행별 값을먼저 구한뒤조건별로평균한다. 평균vx/vy RMSE를구한뒤합치면다른양이므로그렇게계산하지 않는다. 선속도두축만합쳐단위는m/s이며yaw의rad/s를섞지않는다. 전진/대각/횡이동에서명령성분이다르다는점을반영하고,몸체가정지해명령0.5m/s를못따르면약0.5m/s의오차가나온다. 전체성능곡선의vx/vy/yaw/횡오차지표는각각유지한다.

## 버티기 해석

`stall_seconds>0`은30초동안새경로진행0.5m미만인구간이한번이상관찰됐다는뜻이다. 완전한정지나영구실패판정과같지않고,정체후완주한경우도포함한다.

`생존 시행 중 정체 경험 비율 = (생존했고 정체flag가 있었던 시행 수) / (생존 시행 수)`

은높은생존율을해석하는보조통계다. RL calf100%저하에서는Normal185/188=98.40%,Student4/198=2.02%다. 미완주상한까지남은정체로한정하면각각185/188=98.40%,3/198=1.52%다. 분모를전체시행216회와혼동하지않는다. 이것을새정책의주보상이나유일한성능점수로사용하지않는다.

생존율은낙상0/비낙상1의평균이다. 정체·저속으로조기종료하지않았다. 오차는고장후낙상/완주/상한전유효구간의시행별RMSE평균이다. 유효시간과생존율을함께읽는다. 평가시드3개는독립학습3회가아니다.
