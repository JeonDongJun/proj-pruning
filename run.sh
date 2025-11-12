#!/bin/bash
EPOCHS_DENSE=20
EPOCHS_PRUNE=1
BATCH_SIZE=512
SPARSITIES=(0.3 0.5 0.7 0.9)

python main.py --generate_plots --sparsities ${SPARSITIES[@]} --method all --epochs_dense $EPOCHS_DENSE --epochs_prune $EPOCHS_PRUNE --batch_size $BATCH_SIZE
echo "All experiments completed!"
