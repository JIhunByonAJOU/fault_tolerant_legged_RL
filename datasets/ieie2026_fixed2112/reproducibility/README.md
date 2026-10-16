# 재현 근거 보완

- `runtime_evidence.json`: 원래 18개 배치 로그의 경로·SHA-256과 실제 런타임 헤더. 현재 소프트웨어 버전을 소급한 기록이 아니다.
- `config_reference/`: 최초 데이터 게시 커밋 `6cde37b`에서 읽은 상속 설정 보완본. 각 파일 hash는 `reference_manifest.json`에 있다. 당시 확장된 runtime cfg 전체를 저장했다는 뜻은 아니며, 수집기의 명시적 override와 실제 적용 상태 파일을 우선한다.
- `reconstruct_terrain.py`: 동결한 지형 생성 코드를 사용해 원래 시드의 높이맵을 CPU에서 재구성한다. 세 높이 배열의 hash를 원래 평가 기록과 비교한 후 저장한다. 보행 시뮬레이터나 학습은 만들지 않는다.

## 원래 보행 수집 명령 구조

동결한 `source_snapshot/legged_gym/scripts/run_course_campaign.py`는 각 배치를 다음 형태로 호출했다. 아래는 출처 설명이며 현재 데이터 폴더를 지정해 재실행하라는 명령이 아니다. 원래 프로토콜의 이전 비율 탐색 계획을 그대로 실행하면 현재 최종 구성과 다른 작업이 시작될 수 있다.

```bash
python -u -m legged_gym.scripts.collect_course_batch \
  --campaign <original-campaign-directory> \
  --job <original-job-specification.json> \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

같은 체크포인트·소스·의존성·시드가 필요하다. 체크포인트 바이너리와 Isaac Gym 배포본은 저장소에 넣지 않았다. Windows에서 원자료 분석과 그림 재생성에는 이 환경이 필요 없다.

## 높이맵만 다시 검증하는 Ubuntu 명령

원래 WIM 환경을 가진 이 머신의 저장소 루트 기준이다.

```bash
PATH=/home/jihun/Capstone2/miniconda3/envs/WIM/bin:$PATH \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/jihun/legged_gym:/home/jihun/Capstone2/isaacgym/python \
/home/jihun/Capstone2/miniconda3/envs/WIM/bin/python \
  datasets/ieie2026_fixed2112/reproducibility/reconstruct_terrain.py
```

다른 머신은 설치 경로를 맞춘다. 결과는 `terrain/heightfield_s1101.npz` 등이다. 이 파일은 생성 모델의 임의 시각화가 아니라, 원래 실행 시 높이 배열 hash와 일치하는 재구성 자료다. Windows에서는 NumPy만으로 바로 읽을 수 있다.
