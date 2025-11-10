import torch
import torch.nn as nn
import numpy as np
import copy


def magnitude_pruning(model, sparsity, device=None):
    """
    Method A: Magnitude-based Pruning
    가장 작은 절댓값을 가진 가중치를 제거
    """
    model = copy.deepcopy(model)
    
    # 디바이스 정보 유지
    if device is None:
        # 모델의 첫 번째 파라미터에서 디바이스 확인
        device = next(model.parameters()).device
    
    model = model.to(device)
    
    # 모든 파라미터 수집
    all_weights = []
    for param in model.parameters():
        if len(param.data.shape) >= 2:  # Conv, Linear 레이어만
            all_weights.append(param.data.abs().flatten())
    
    if not all_weights:
        return model
    
    # 전체 가중치의 threshold 계산
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity)
    
    # Pruning 적용
    for param in model.parameters():
        if len(param.data.shape) >= 2:
            mask = param.data.abs() > threshold
            param.data *= mask.float()
    
    return model


def obd_pruning(model, train_loader, device, sparsity, num_batches=10):
    """
    Method B: OBD (Optimal Brain Damage) Pruning
    헤시안의 대각 성분을 직접 계산하여 pruning
    Saliency = 0.5 * H_ii * w_i^2
    """
    model = copy.deepcopy(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 헤시안 대각 성분을 저장할 딕셔너리
    hessian_diag = {}
    for name, param in model.named_parameters():
        if len(param.data.shape) >= 2:  # Conv, Linear 레이어만
            hessian_diag[name] = torch.zeros_like(param.data)
    
    model.to(device)
    batch_count = 0
    
    # 각 배치에 대해 헤시안 대각 성분 계산
    for batch in train_loader:
        if batch_count >= num_batches:
            break
        
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        # 각 파라미터에 대해 헤시안 대각 성분 계산
        for name, param in model.named_parameters():
            if name not in hessian_diag:
                continue
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 첫 번째 gradient 계산 (create_graph=True로 설정하여 2차 미분 가능하게)
            grad = torch.autograd.grad(
                loss, param, create_graph=True, retain_graph=True
            )[0]
            
            if grad is not None:
                # 헤시안 대각 성분 계산: H_ii = ∂²L/∂w_i²
                # 각 파라미터 원소에 대해 개별적으로 계산
                param_flat = param.data.flatten()
                grad_flat = grad.flatten()
                hessian_flat = torch.zeros_like(param_flat)
                
                # 각 원소에 대해 헤시안 대각 성분 계산
                for i in range(param_flat.numel()):
                    # grad[i]에 대한 gradient를 계산하면 헤시안 대각 성분
                    hessian_elem = torch.autograd.grad(
                        grad_flat[i], param, retain_graph=True
                    )[0]
                    
                    if hessian_elem is not None:
                        hessian_flat[i] = hessian_elem.flatten()[i]
                
                # 누적
                hessian_diag[name] += hessian_flat.reshape(param.data.shape)
        
        batch_count += 1
    
    # 평균 계산
    if batch_count > 0:
        for name in hessian_diag:
            hessian_diag[name] /= batch_count
    
    # Saliency 계산: S_i = 0.5 * H_ii * w_i^2
    all_saliencies = []
    for name, param in model.named_parameters():
        if name in hessian_diag:
            # 0.5는 상수이므로 순위에 영향 없어 생략 가능
            saliency = hessian_diag[name] * param.data.pow(2)
            all_saliencies.append(saliency.flatten())
    
    # Saliency가 낮은 가중치 제거
    if all_saliencies:
        all_saliencies = torch.cat(all_saliencies)
        threshold = torch.quantile(all_saliencies, sparsity)
        
        # Pruning 적용
        for name, param in model.named_parameters():
            if name in hessian_diag:
                saliency = hessian_diag[name] * param.data.pow(2)
                mask = saliency > threshold
                param.data *= mask.float()
    
    return model


def lottery_ticket_pruning(model, initial_weights, sparsity, device=None):
    """
    Method C: Lottery Ticket Pruning
    초기 가중치를 저장하고, magnitude pruning 후 초기 가중치로 복원
    """
    model = copy.deepcopy(model)
    
    # 디바이스 정보 유지
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    
    # Magnitude 기반으로 마스크 생성
    all_weights = []
    for param in model.parameters():
        if len(param.data.shape) >= 2:
            all_weights.append(param.data.abs().flatten())
    
    if not all_weights:
        return model
    
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity)
    
    # 마스크 생성 및 초기 가중치로 복원
    for name, param in model.named_parameters():
        if len(param.data.shape) >= 2 and name in initial_weights:
            mask = param.data.abs() > threshold
            # Lottery Ticket: 마스크된 부분만 초기 가중치로 복원
            # initial_weights도 같은 디바이스로 이동
            init_weight = initial_weights[name].clone().to(device)
            param.data = init_weight * mask.float()
    
    return model


def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.data.numel()
        zero_params += (param.data == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0

