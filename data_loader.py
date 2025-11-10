import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']  # PIL Image
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
        
        return {'pixel_values': image, 'label': label}


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'pixel_values': pixel_values, 'label': labels}


def get_cifar10_loaders(batch_size=128, num_workers=0):
    dataset = load_dataset("cifar10")

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CIFAR10Dataset(dataset['train'], transform=train_transform)
    test_dataset = CIFAR10Dataset(dataset['test'], transform=test_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, _ = get_cifar10_loaders(batch_size=4)
    batch = next(iter(train_loader))
    print("Batch images shape:", batch['pixel_values'].shape)
    print("Batch labels shape:", batch['label'].shape)