"""
Main file for the CIFAR-10 pruning experiment
Author: JeonDongJun
Date: 2025-11-11
Reference: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import argparse
import torch
import numpy as np
import random
import os
from model import create_model
from data_loader import get_cifar10_loaders
from trainer import SimpleTrainer
from pruning import magnitude_pruning, obd_pruning, lottery_ticket_pruning, calculate_sparsity


def train_dense_model(train_loader, test_loader, device, epochs):
    model = create_model()
    initial_weights = {name: param.data.clone() for name, param in model.named_parameters()}
    
    trainer = SimpleTrainer(model, device)
    best_acc = trainer.train(train_loader, test_loader, epochs)
    
    return model, best_acc, initial_weights


def apply_pruning_and_evaluate(model, train_loader, test_loader, device, method, sparsity, epochs=50, initial_weights=None):    
    # Pruning 적용 후 fine-tuning 및 평가

    if method == 'magnitude':
        pruned_model = magnitude_pruning(model, sparsity, device=device)
    elif method == 'obd':
        pruned_model = obd_pruning(model, train_loader, device, sparsity, num_batches=10)
    elif method == 'lottery_ticket':
        pruned_model = lottery_ticket_pruning(model, initial_weights, sparsity, device=device)
    
    actual_sparsity = calculate_sparsity(pruned_model)
    print(f"Actual Sparsity: {actual_sparsity:.2%}")
    
    # Fine-tuning
    trainer = SimpleTrainer(pruned_model, device, lr=0.01)
    best_acc = trainer.train(train_loader, test_loader, epochs)
    
    return pruned_model, best_acc, actual_sparsity


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Pruning Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs_dense', type=int, default=200, help='Epochs for dense training')
    parser.add_argument('--epochs_prune', type=int, default=50, help='Epochs for fine-tuning after pruning')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity (0.0-1.0)')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['magnitude', 'obd', 'lottery_ticket', 'all'],
                       help='Pruning method to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    dense_model, dense_acc, initial_weights = train_dense_model(
        train_loader, test_loader, device, args.seed, args.epochs_dense
    )
    print(f"\nDense Model Accuracy: {dense_acc:.2f}%")
        
    methods = ['magnitude', 'obd', 'lottery_ticket'] if args.method == 'all' else [args.method]    
    results = {}
    for method in methods:
        pruned_model, pruned_acc, actual_sparsity = apply_pruning_and_evaluate(
            dense_model, train_loader, test_loader, device, method, args.sparsity, 
            args.epochs_prune, initial_weights if method == 'lottery_ticket' else None
        )
        results[method] = {
            'pruned_model': pruned_model,
            'accuracy': pruned_acc,
            'sparsity': actual_sparsity
        }
        print(f"\n{method} - Accuracy: {pruned_acc:.2f}%, Sparsity: {actual_sparsity:.2%}")
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("Results Summary")
    print(f"{'='*50}")
    print(f"Dense Model: {dense_acc:.2f}%")
    for method, result in results.items():
        print(f"{method}: {result['accuracy']:.2f}% (Sparsity: {result['sparsity']:.2%})")

if __name__ == '__main__':
    main()

