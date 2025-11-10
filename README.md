# CIFAR-10 Pruning 프로젝트

3가지 경량화(Pruning) 방법을 적용한 CIFAR-10 분류 모델 학습 프로젝트입니다.

## 프로젝트 구조

```
.
├── data_loader.py      # HuggingFace datasets를 사용한 CIFAR-10 데이터 로더
├── model.py            # ResNet18 모델 아키텍처
├── pruning.py          # 3가지 Pruning 방법 구현
├── trainer.py          # 심플한 학습 및 평가 클래스
├── main.py             # 메인 실행 파일
├── requirements.txt    # 필요한 패키지 목록
└── run.sh              # 실험 실행 스크립트
```

## Pruning 방법

1. **Magnitude Pruning**: 가장 작은 절댓값을 가진 가중치를 제거
2. **OBD (Optimal Brain Damage)**: Loss의 gradient magnitude를 기반으로 중요도 계산하여 pruning
3. **Lottery Ticket**: 초기 가중치를 저장하고, magnitude pruning 후 초기 가중치로 복원

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 실행

```bash
# Dense 모델 학습
python main.py --seed 42 --epochs_dense 200

# 특정 pruning 방법 적용
python main.py --seed 42 --method magnitude --sparsity 0.5
python main.py --seed 42 --method obd --sparsity 0.5
python main.py --seed 42 --method lottery_ticket --sparsity 0.5

# 모든 pruning 방법 테스트
python main.py --seed 42 --method all --sparsity 0.5
```

### 주요 옵션

- `--seed`: 랜덤 시드 (기본값: 42)
- `--epochs_dense`: Dense 모델 학습 에폭 수 (기본값: 200)
- `--epochs_prune`: Pruning 후 fine-tuning 에폭 수 (기본값: 50)
- `--sparsity`: 목표 sparsity 비율 (0.0-1.0, 기본값: 0.5)
- `--method`: Pruning 방법 선택 (`magnitude`, `obd`, `lottery_ticket`, `all`)
- `--batch_size`: 배치 크기 (기본값: 128)

### 전체 실험 실행

```bash
bash run.sh
```

## 예시

```bash
# Seed 42로 magnitude pruning (sparsity 0.5)
python main.py --seed 42 --method magnitude --sparsity 0.5 --epochs_dense 200 --epochs_prune 50

# Seed 123으로 OBD pruning
python main.py --seed 123 --method obd --sparsity 0.7

# Seed 456으로 Lottery Ticket pruning
python main.py --seed 456 --method lottery_ticket --sparsity 0.5

# 모든 방법 테스트
python main.py --seed 42 --method all --sparsity 0.7
```

