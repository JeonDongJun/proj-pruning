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
import time
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
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


def count_parameters(model):
    """모델의 파라미터 수를 계산 (M 단위)"""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6  # Million 단위

def get_model_size_mb(model):
    """모델 크기를 계산 (MB 단위)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_latency(model, test_loader, device, num_samples=100):
    """추론 지연시간 측정 (ms/image)"""
    model.eval()
    model.to(device)
    
    # Warm-up
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:
                break
            inputs = batch['pixel_values'].to(device)
            _ = model(inputs)
    
    # 측정
    latencies = []
    with torch.no_grad():
        sample_count = 0
        for batch in test_loader:
            if sample_count >= num_samples:
                break
            inputs = batch['pixel_values'].to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(inputs)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            batch_size = inputs.size(0)
            latency_per_image = (end_time - start_time) / batch_size * 1000  # ms
            latencies.extend([latency_per_image] * batch_size)
            sample_count += batch_size
    
    return np.mean(latencies)

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
    
    # 통계 계산
    num_params = count_parameters(pruned_model)
    model_size = get_model_size_mb(pruned_model)
    latency = measure_inference_latency(pruned_model, test_loader, device)
    
    return pruned_model, best_acc, actual_sparsity, num_params, model_size, latency


def run_experiment_with_seeds(sparsities, methods, seeds, epochs_dense, epochs_prune, batch_size, device):
    all_results = {method: {sparsity: [] for sparsity in sparsities} for method in methods}
    all_results['dense'] = {
        'accuracy': [],
        'sparsity': [],
        'params': [],
        'size': [],
        'latency': []
    }
    
    for seed in seeds:
        torch.manual_seed(seed)
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
        
        dense_model, dense_acc, initial_weights = train_dense_model(
            train_loader, test_loader, device, epochs_dense
        )
        
        dense_params = count_parameters(dense_model)
        dense_size = get_model_size_mb(dense_model)
        dense_latency = measure_inference_latency(dense_model, test_loader, device)
        
        print(f"Dense Model Accuracy: {dense_acc:.2f}%")
        print(f"  Params: {dense_params:.2f}M, Size: {dense_size:.2f}MB, Latency: {dense_latency:.2f}ms")
        
        all_results['dense']['accuracy'].append(dense_acc)
        all_results['dense']['sparsity'].append(0.0)
        all_results['dense']['params'].append(dense_params)
        all_results['dense']['size'].append(dense_size)
        all_results['dense']['latency'].append(dense_latency)
        
        for method in methods:
            for sparsity in sparsities:
                pruned_model, pruned_acc, actual_sparsity, num_params, model_size, latency = apply_pruning_and_evaluate(
                    dense_model, train_loader, test_loader, device, method, sparsity,
                    epochs_prune, initial_weights if method == 'lottery_ticket' else None
                )
                
                all_results[method][sparsity].append({
                    'accuracy': pruned_acc,
                    'sparsity': actual_sparsity,
                    'params': num_params,
                    'size': model_size,
                    'latency': latency
                })
                print(f"Method: {method}, Sparsity: {sparsity:.2f}")
                print(f"Result: Acc={pruned_acc:.2f}%, Sparsity={actual_sparsity:.2%}, Params={num_params:.2f}M, Size={model_size:.2f}MB, Latency={latency:.2f}ms")

    return all_results

def calculate_ci(data, confidence=0.95):
    if len(data) == 0:
        return 0, 0
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, h

def plot_tradeoff_curve(all_results, methods, sparsities, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    method_colors = {
        'magnitude': 'blue',
        'obd': 'red',
        'lottery_ticket': 'green'
    }
    method_labels = {
        'magnitude': 'Magnitude Pruning',
        'obd': 'OBD Pruning',
        'lottery_ticket': 'Lottery Ticket'
    }
    
    for method in methods:
        accuracies = []
        sparsity_values = []
        ci_lowers = []
        ci_uppers = []
        
        dense_accs = all_results['dense']['accuracy']
        mean_acc, ci = calculate_ci(dense_accs)
        accuracies.append(mean_acc)
        sparsity_values.append(0.0)
        ci_lowers.append(mean_acc - ci)
        ci_uppers.append(mean_acc + ci)
        
        for sparsity in sorted(sparsities):
            results = all_results[method][sparsity]
            accs = [r['accuracy'] for r in results]
            mean_acc, ci = calculate_ci(accs)
            accuracies.append(mean_acc)
            sparsity_values.append(sparsity)
            ci_lowers.append(mean_acc - ci)
            ci_uppers.append(mean_acc + ci)
        
        if len(accuracies) > 0:
            plt.plot(sparsity_values, accuracies, 'o-', color=method_colors.get(method, 'black'),
                    label=method_labels.get(method, method), linewidth=2, markersize=8)
            plt.fill_between(sparsity_values, ci_lowers, ci_uppers, 
                           color=method_colors.get(method, 'black'), alpha=0.2)
    
    plt.xlabel('Sparsity', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Tradeoff Curve: Test Accuracy vs Sparsity', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff_curve.png'), dpi=300)
    plt.close()

def create_efficiency_table(all_results, methods, sparsities, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    table_data = []
    
    # Dense 모델
    dense_accs = all_results['dense']['accuracy']
    dense_params = all_results['dense']['params']
    dense_sizes = all_results['dense']['size']
    dense_latencies = all_results['dense']['latency']
    
    mean_acc, _ = calculate_ci(dense_accs)
    mean_params = np.mean(dense_params)
    mean_size = np.mean(dense_sizes)
    mean_latency = np.mean(dense_latencies)
    
    table_data.append({
        'model': 'ResNet18',
        'sparsity': 0.00,
        'pruning_method': 'N/A (dense model)',
        'top1': f"{mean_acc:.1f}",
        'params_M': f"{mean_params:.2f}",
        'size_MB': f"{mean_size:.2f}",
        'lat_ms': f"{mean_latency:.1f}"
    })
    
    for method in methods:
        method_name = {
            'magnitude': 'magnitude pruning',
            'obd': 'OBD pruning',
            'lottery_ticket': 'lottery ticket pruning'
        }.get(method, method)
        
        for sparsity in sorted(sparsities):
            if sparsity not in all_results[method]:
                continue
            results = all_results[method][sparsity]
            
            accs = [r['accuracy'] for r in results]
            params = [r['params'] for r in results]
            sizes = [r['size'] for r in results]
            latencies = [r['latency'] for r in results]
            
            mean_acc, _ = calculate_ci(accs)
            mean_params = np.mean(params)
            mean_size = np.mean(sizes)
            mean_latency = np.mean(latencies)
            
            table_data.append({
                'model': 'ResNet18',
                'sparsity': f"{sparsity:.2f}",
                'pruning_method': method_name,
                'top1': f"{mean_acc:.1f}",
                'params_M': f"{mean_params:.2f}",
                'size_MB': f"{mean_size:.2f}",
                'lat_ms': f"{mean_latency:.1f}"
            })
    
    df = pd.DataFrame(table_data)
    df.columns = ['model', 'sparsity', 'pruning method', 'top1', 'params_M', 'size_MB', 'lat_ms']
    
    md_path = os.path.join(output_dir, 'efficiency_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(df.to_markdown(index=False))
    
    csv_path = os.path.join(output_dir, 'efficiency_table.csv')
    df.to_csv(csv_path, index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Pruning Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (single run)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], 
                       help='Random seeds for multiple runs (for CI calculation)')
    parser.add_argument('--epochs_dense', type=int, default=200, help='Epochs for dense training')
    parser.add_argument('--epochs_prune', type=int, default=50, help='Epochs for fine-tuning after pruning')
    parser.add_argument('--sparsity', type=float, default=0.5, help='Target sparsity (0.0-1.0) - single run')
    parser.add_argument('--sparsities', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Target sparsities for tradeoff curve')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['magnitude', 'obd', 'lottery_ticket', 'all'],
                       help='Pruning method to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--generate_plots', action='store_true', 
                       help='Generate tradeoff curve and efficiency table')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    methods = ['magnitude', 'obd', 'lottery_ticket'] if args.method == 'all' else [args.method]
    
    if args.generate_plots:
        all_results = run_experiment_with_seeds(
            args.sparsities, methods, args.seeds, 
            args.epochs_dense, args.epochs_prune, args.batch_size, device
        )
        
        # Tradeoff curve 생성
        plot_tradeoff_curve(all_results, methods, args.sparsities, args.output_dir)
        
        # Efficiency table 생성
        create_efficiency_table(all_results, methods, args.sparsities, args.output_dir)
        
    else:
        torch.manual_seed(args.seed)
        
        train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
        dense_model, dense_acc, initial_weights = train_dense_model(
            train_loader, test_loader, device, args.epochs_dense
        )
        
        # Dense model 통계 계산
        dense_params = count_parameters(dense_model)
        dense_size = get_model_size_mb(dense_model)
        dense_latency = measure_inference_latency(dense_model, test_loader, device)
        
        print(f"\nDense Model Accuracy: {dense_acc:.2f}%")
        print(f"  Params: {dense_params:.2f}M, Size: {dense_size:.2f}MB, Latency: {dense_latency:.2f}ms")
        
        results = {}
        for method in methods:
            pruned_model, pruned_acc, actual_sparsity, num_params, model_size, latency = apply_pruning_and_evaluate(
                dense_model, train_loader, test_loader, device, method, args.sparsity,
                args.epochs_prune, initial_weights if method == 'lottery_ticket' else None
            )
            results[method] = {
                'accuracy': pruned_acc,
                'sparsity': actual_sparsity,
                'params': num_params,
                'size': model_size,
                'latency': latency
            }
            print(f"\n{method} - Accuracy: {pruned_acc:.2f}%, Sparsity: {actual_sparsity:.2%}")
            print(f"  Params: {num_params:.2f}M, Size: {model_size:.2f}MB, Latency: {latency:.2f}ms")
        
        print(f"\n{'='*50}")
        print("Results Summary")
        print(f"{'='*50}")
        print(f"Dense Model: {dense_acc:.2f}% (Params: {dense_params:.2f}M, Size: {dense_size:.2f}MB, Latency: {dense_latency:.2f}ms)")
        for method, result in results.items():
            print(f"{method}: {result['accuracy']:.2f}% (Sparsity: {result['sparsity']:.2%}, Params: {result['params']:.2f}M, Size: {result['size']:.2f}MB, Latency: {result['latency']:.2f}ms)")

if __name__ == '__main__':
    main()

