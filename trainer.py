"""심플한 Trainer 클래스"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SimpleTrainer:
    """간단한 학습 및 평가 클래스"""
    
    def __init__(self, model, device, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200
        )
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            inputs = batch['pixel_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        self.scheduler.step()
        return running_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, test_loader):
        """평가"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                inputs = batch['pixel_values'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, train_loader, test_loader, epochs=200):
        """전체 학습 과정"""
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            train_loss, train_acc = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'Best accuracy: {best_acc:.2f}%')
        
        return best_acc

