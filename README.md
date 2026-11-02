# Agibot Toy2Box — 행동 차원 축소를 통한 모방학습

IsaacLab v2.3.0 기반 로보틱스 시뮬레이션 수업 프로젝트. 사람 시범100개에서 두 회전 명령을 제한한 성공 Mimic1000개를 생성하고 BC-RNN 정책을 학습했습니다.

## 최종 모델과 실제 결과

| 학습 시드 | 검증 시드 | 성공/평가 | 성공률 |
|---|---|---|---|
|101|42|72/100|72%|
|111|42|70/100|70%|

두 모델 평균71%, 시드 간 모집단 SD1%p(n=2). 각 모델은 최종 epoch2000, horizon800으로 평가했습니다. 제출 지침에 따라 가중치는 **101 시드의 최종 모델1개만** 포함합니다. 111 시드의 원본 결과 JSON은 `submission/`에 보존했습니다. 알려진 검증 시드42에서의 결과이며 미공개 시드 성능을 보장하지 않습니다.

## 실행 환경

테스트 환경: Ubuntu22.04, Python3.11, IsaacSim5.1.0.0, IsaacLab v2.3.0 코드, torch2.7.0+cu128, robomimic0.4.0, gymnasium1.2.0. NVIDIA GPU 및 IsaacLab/IsaacSim 자산 접근이 필요합니다.

IsaacSim이 준비된 Python/Conda 환경에서 이 프로젝트를 복제한 뒤 해당 프로젝트의 패키지를 설치합니다. 이 저장소의 환경 코드를 사용해야 하므로 다른 IsaacLab 체크아웃이 editable 설치되어 있다면 여기에서 다시 설치합니다.

```bash
conda activate lerobot-arena
./isaaclab.sh -i
python -m pip install -r requirements-agibot.txt
```

기본 IsaacLab 설치 안내·저작권은 [원본 README](README_ISAACLAB_UPSTREAM.md)와 `LICENSE`에 유지했습니다. 라이브러리와 IsaacSim의 설치 경로는 각 컴퓨터에 맞게 준비합니다.

## 평가 명령 — LMS에 함께 제출

프로젝트 최상위 폴더에서 실행합니다.

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint ./checkpoints/final_seed101_epoch2000.pth \
  --num_rollouts 100 --horizon 800 --seed 42 --headless \
  --results_file ./artifacts/eval_seed42.json
```

`--seed`는 평가자가 원하는 값으로 바꿀 수 있습니다. 기존 결과 파일은 덮어쓰지 않으므로 재평가 시 새 `--results_file`을 사용합니다. 가중치만 있으면 평가할 수 있으며 HDF5는 평가에 필요하지 않습니다.

영상 녹화에는 위 명령에 다음 옵션을 추가합니다. 640×480, 원속도20fps로 프레임을 즉시 인코딩합니다.

```bash
--video --video_length 1000 --video_folder ./artifacts/videos/seed42
```

## 5D 행동 표현

원래 환경은7D `[dx,dy,dz,r_x,r_y,r_z,g]`, 최종 정책은5D `[dx,dy,dz,r_y,g]`입니다. `play.py`가 체크포인트 차원을 확인해 이 Agibot 과제의5D 정책에만 어댑터를 적용합니다. 환경에는 `[dx,dy,dz,0,r_y,0,g]`를 전달하고 관측23D는 유지합니다. 호환되는7D 정책은 그대로 실행합니다.

제한한 `r_x,r_z`는 로봇 기준좌표계 X/Z축 주위 상대 회전 명령(rad)입니다. 손목 관절 두 개의 물리적 고정과 구분합니다. 사람 시범은 첫20개부터 품질과 일관성을 고려해 수집했습니다. 최종 실험은 **Mimic 생성 제약과 정책 출력 축소를 함께 바꾼 실험**이므로 두 조치의 독립 효과는 분리되지 않았습니다.

## 증강과 재학습

`submission_data/`에는 동일한 최종 사람 시범100개와 자동 주석본을 Git LFS로 포함합니다. 복제 후 `git lfs install`과 `git lfs pull`을 실행해 실제 HDF5를 받습니다(Git LFS 설치 필요). 체크섬은 `submission/data_manifest.json` 참조. 최종5D 학습 데이터는 용량 때문에 별도 `Agibot_Toy2Box_재학습데이터_20261102.zip`으로 준비했습니다. 이 ZIP을 프로젝트 최상위에서 풀면 `datasets/`가 생성됩니다. [공개 Release에서 재학습 데이터 ZIP 다운로드](https://github.com/Hyunlulu/agibot-toy2box-imitation-learning/releases/tag/training-data-v1).

생성부터 반복하려면 다음 명령을 사용합니다. 병렬 환경16개에서 **성공 시범1000개**가 모일 때까지 생성하고, 노이즈 적용 뒤 실제 실행 명령의 두 회전을0으로 제한합니다. 실패 시범은 학습에 사용하지 않습니다.

```bash
mkdir -p datasets
./isaaclab.sh -p scripts/imitation_learning/agibot_action5/generate_constrained.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --headless --num_envs 16 --generation_num_trials 1000 \
  --input_file ./submission_data/source100_annotated.hdf5 \
  --output_file ./datasets/generated_constrained.hdf5

./isaaclab.sh -p scripts/imitation_learning/agibot_action5/training_data.py \
  --input_file ./datasets/generated_constrained.hdf5 \
  --output_file ./datasets/train_action5.hdf5 --count 1000

./isaaclab.sh -p scripts/imitation_learning/agibot_action5/train_seed.py \
  --seed 101 --dataset ./datasets/train_action5.hdf5
```

111 시드는 마지막 명령의 `--seed 111` 사용. 코드·설정은 `scripts/imitation_learning/agibot_action5/`, 실행 결과와 잠금 파일은 `artifacts/action5/`, 새 학습 체크포인트는 `logs/agibot_submission_training/`에 생성됩니다. 중간 결과나 기존 데이터 파일을 덮어쓰지 않습니다. 재생성을 반복하면 데이터 선택과 시뮬레이션 조건에 따라 생성본과 재학습 결과는 달라질 수 있습니다.

## 검증

```bash
(cd scripts/imitation_learning/robomimic && python -m unittest test_action_adapter test_action5_submission -q)
(cd scripts/imitation_learning/agibot_action5 && python -m unittest test_training_data test_policy_rollout test_generation_help test_generation_cli -q)
```

단위 검사25개: 5D↔7D 슬롯, 제거 회전0 보장, 관측·상태 보존, 잘못된 차원 거부, 임의 평가 시드 기록, 결과 파일 보존, 성공·유한값 데이터만 변환. GPU 실행 점검 결과는 `submission/verification.json`에서 확인합니다.

## 발표 자료와 출처

`submission/`의 PPT/PDF 및 평가 명령어 텍스트 참조. PPT 마지막 두 장에는 기존100회씩 평가한 전체 영상이 들어 있습니다. 별도의 새 평가 표본으로 합산하지 않습니다. 라이브러리 전체를 포함해 실행 프로젝트로 제출하되, 다른 Ant 과제의 추가 수정은 포함하지 않았습니다.

기반: [cailab-hy/IsaacLab_RS](https://github.com/cailab-hy/IsaacLab_RS), IsaacLab 원본 commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`. 기반 라이선스 파일과 소스 저작권 표기는 보존했습니다.
