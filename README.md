# CIFAR-10 Pruning 프로젝트

### 전체 실험 실행

```bash
bash run.sh
```

```bash
# magnitude pruning (sparsity 0.5)
python main.py --seed 42 --method magnitude --sparsity 0.5 --epochs_dense 200 --epochs_prune 50

# OBD pruning
python main.py --seed 42 --method obd --sparsity 0.7

# Lottery Ticket pruning
python main.py --seed 42 --method lottery_ticket --sparsity 0.5

# 모든 방법 테스트
python main.py --seed 42 --method all --sparsity 0.7
```

