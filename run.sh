#!/bin/bash

# CIFAR-10 Pruning 실험 실행 스크립트

# 기본 설정
SEED=42
EPOCHS_DENSE=200
EPOCHS_PRUNE=50
BATCH_SIZE=128
SPARSITY=0.5

# Dense 모델 학습
echo "Training Dense Model..."
python main.py --seed $SEED --epochs_dense $EPOCHS_DENSE --batch_size $BATCH_SIZE --method magnitude --sparsity 0.0

# Pruning 실험 (3가지 방법, 여러 sparsity 레벨)
SPARSITIES=(0.3 0.5 0.7 0.9)
METHODS=("magnitude" "obd" "lottery_ticket")

for method in "${METHODS[@]}"; do
    for sparsity in "${SPARSITIES[@]}"; do
        echo "Running $method pruning with sparsity $sparsity..."
        python main.py --seed $SEED --epochs_dense $EPOCHS_DENSE --epochs_prune $EPOCHS_PRUNE \
            --batch_size $BATCH_SIZE --method $method --sparsity $sparsity
    done
done

echo "All experiments completed!"

