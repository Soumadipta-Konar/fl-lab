import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_mnist(data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_x = train_set.data.float().unsqueeze(1) / 255.0
    train_x = (train_x - 0.1307) / 0.3081
    train_y = train_set.targets

    test_x = test_set.data.float().unsqueeze(1) / 255.0
    test_x = (test_x - 0.1307) / 0.3081
    test_y = test_set.targets

    return (train_x, train_y), (test_x, test_y)

def evaluate(model, x, y, batch_size=512, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    correct = 0
    num_samples = len(y)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bx = x[i:i+batch_size].to(device)
            by = y[i:i+batch_size].to(device)
            out = model(bx)
            total_loss += criterion(out, by).item()
            pred = out.argmax(dim=1)
            correct += (pred == by).sum().item()

    avg_loss = total_loss / num_samples
    acc = correct / num_samples
    return avg_loss, acc
