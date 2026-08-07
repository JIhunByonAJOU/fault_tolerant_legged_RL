# `legged_ws` WIM 실행 환경

- 구성일: 2026-08-07
- 목적: 현재 `/home/jihun/legged_gym`의 WIM(`legged_gym`)을 NVIDIA RTX 4090에서 Isaac Gym으로 실행
- Conda prefix: `/home/jihun/Capstone2/miniconda3/envs/legged_ws`
- 설치 방식: 기존 검증 환경 `WIM`을 clone한 뒤 현재 저장소를 editable install

## 핵심 버전과 출처

| 구성요소 | 버전·위치 | 비고 |
|---|---|---|
| Python | 3.8.20 | Isaac Gym Preview 4의 `<3.9` 조건 충족 |
| Isaac Gym | 1.0 Preview 4 | `/home/jihun/Capstone2/isaacgym/python`, editable install |
| PyTorch | 1.13.1+cu117 | CUDA runtime 11.7, RTX 4090 인식 확인 |
| torchvision | 0.14.1+cu117 | PyTorch 조합 고정 |
| rsl_rl | 1.0.2, commit `2ad79cf0caa85b91721abfe358105f869a784121` | `/home/jihun/Capstone2/rsl_rl`, official tag `v1.0.2` |
| legged_gym | 1.0.0, current workspace editable | `/home/jihun/legged_gym` |
| NumPy / SciPy | 1.23.5 / 1.10.1 | terrain generator import 확인 |
| GPU | NVIDIA GeForce RTX 4090 24 GB | GPU PhysX와 GPU pipeline 확인 |

## 활성화

현재 login shell에는 Conda hook이 자동 등록되어 있지 않다. 새 shell에서 다음처럼 활성화한다.

```bash
source /home/jihun/Capstone2/miniconda3/etc/profile.d/conda.sh
conda activate legged_ws
cd /home/jihun/legged_gym
```

Isaac Gym은 PyTorch보다 먼저 import해야 한다. WIM의 scripts는 이미 이 순서를 따른다.

## 재구성 명령

```bash
/home/jihun/Capstone2/miniconda3/bin/conda create -y -n legged_ws --clone WIM
/home/jihun/Capstone2/miniconda3/bin/conda run -n legged_ws \
  python -m pip install --no-deps -e /home/jihun/legged_gym
```

Isaac Gym과 `rsl_rl`은 다음 local source를 editable install한 상태다.

```text
/home/jihun/Capstone2/isaacgym/python
/home/jihun/Capstone2/rsl_rl
```

## 검증 결과

1. Import 검증
   - `isaacgym`, `torch`, `torchvision`, `numpy`, `scipy`, `rsl_rl`, `legged_gym` import 성공
   - `legged_gym`이 `/home/jihun/legged_gym/legged_gym/__init__.py`를 가리키는 것 확인
   - 등록 task: `a1`, `anymal_b`, `anymal_c_flat`, `anymal_c_rough`, `cassie`
2. CUDA 검증
   - PyTorch CUDA available: `True`
   - GPU: `NVIDIA GeForce RTX 4090`
   - Isaac Gym: GPU PhysX, physics device `cuda:0`, GPU pipeline enabled
3. Flat A1 smoke
   - 8 env, plane, 10 control steps
   - observation `(8, 48)`, reward finite
4. Rough A1 smoke
   - 8 env, trimesh, 축소한 3×5 terrain grid, 3 control steps
   - observation `(8, 235)`, reward finite
5. PPO integration smoke
   - 16 env, plane, rollout 4 steps, PPO learning iteration 1회
   - 총 64 transitions, actor/critic forward·backward 및 checkpoint 저장 성공
   - `PPO_SMOKE_OK` 확인; 임시 로그만 사용했고 repository log에는 쓰지 않음

위 smoke는 환경/physics 경로 확인용이며 policy 학습 성공을 뜻하지 않는다. 첫 실제 학습은 `Saving the Limping` 환경 구현 뒤 64 env gate에서 시작한다.

## 실행 예

기존 WIM A1 학습 경로는 다음과 같다.

```bash
python legged_gym/scripts/train.py --task=a1 --headless
```

`Saving the Limping` task 이름과 구현은 아직 추가되지 않았다. 구현 후 `a1_limping_base`와 `a1_limping_failure`를 별도로 등록한다.
