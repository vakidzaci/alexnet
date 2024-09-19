import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Import AlexNet model (previously defined)
from model import AlexNet  # Assuming the AlexNet model is in alexnet.py

# Define the path to Tiny ImageNet dataset
DATA_DIR = './data'

# Data transforms (data augmentation and normalization)
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])

transform_val = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])

# Load Tiny ImageNet Dataset
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# Initialize AlexNet model
model = AlexNet(num_classes=200)  # Tiny ImageNet has 200 classes

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# Training Function
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward Pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track progress
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(
                f'Epoch {epoch}, Batch {batch_idx}, Loss: {running_loss / (batch_idx + 1):.3f}, Accuracy: {100. * correct / total:.3f}%')

    return running_loss / len(train_loader), 100. * correct / total


# Validation Function
def validate(epoch):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Validation Loss: {val_loss / len(val_loader):.3f}, Validation Accuracy: {100. * correct / total:.3f}%')
    return val_loss / len(val_loader), 100. * correct / total


# Main training loop
for epoch in range(300):  # Train for 50 epochs
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = validate(epoch)
    scheduler.step()

    # Save model checkpoint
    torch.save(model.state_dict(), f'./alexnet_epoch_{epoch}.pth')

print('Training finished!')
