#!/bin/bash

# CIFAR-10 Pruning 실험 실행 스크립트

# 기본 설정
SEED=42
EPOCHS_DENSE=50
EPOCHS_PRUNE=10
BATCH_SIZE=256

# Pruning 실험
SPARSITIES=(0.3 0.5 0.7 0.9)
python main.py --generate_plots --seeds $SEED --sparsities ${SPARSITIES[@]} --method all --epochs_dense $EPOCHS_DENSE --epochs_prune $EPOCHS_PRUNE --batch_size $BATCH_SIZE
echo "All experiments completed!"
