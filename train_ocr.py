import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from student import SimpleCRAFT  # Ensure this is implemented correctly
from easyocr.craft import CRAFT
from collections import OrderedDict
import os
from PIL import Image
from loss import CombinedLoss
from mydatasets import CDIPDataset
from torch.utils.data import random_split
# Define your custom dataset

# Transformations for the input data
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

dataset = CDIPDataset(root_dir='/home/vakidzaci/Downloads/images.a.a/imagesa/a/a', transform=data_transforms)
# Define the sizes for train and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True, num_workers=4)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
student_model = SimpleCRAFT().to(device)
student_model.train()
print(sum(dict((p.data_ptr(), p.numel()) for p in student_model.parameters()).values()))
teacher_model = CRAFT().to(device)
teacher_model.load_state_dict(copyStateDict(torch.load("/home/vakidzaci/.EasyOCR/model/craft_mlt_25k.pth", map_location=device)))
teacher_model.eval()

# Loss function and optimizer
# criterion = nn.MSELoss()
criterion = CombinedLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

def save_model(model, model_name):
    torch.save(model)
# Training function
import time
def train_model(num_epochs):
    student_model.train()
    print(student_model)
    best_val_loss = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        start = time.time()
        for images in train_loader:
            images = images.to(device)

            with torch.no_grad():
                teacher_outputs,_ = teacher_model(images)
                # print(teacher_outputs)
                teacher_outputs = torch.sigmoid(teacher_outputs)

            student_outputs,_ = student_model(images)
            # print(student_outputs)
            student_outputs = torch.sigmoid(student_outputs)
            loss = criterion(student_outputs, teacher_outputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            # Validation loss
        print(running_loss)
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                teacher_outputs,_ = teacher_model(images)
                teacher_outputs = torch.sigmoid(teacher_outputs)
                student_outputs,_ = student_model(images)
                student_outputs = torch.sigmoid(student_outputs)
                loss = criterion(student_outputs, teacher_outputs)
                # print(teacher_outputs[0].size/path/to/cdip_dataset(), student_outputs[0].size())
                val_loss += loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), f'checkpoints/best_model_epoch_{epoch+1}_val_loss_{val_loss}.pth')
            print(f'Saved new best model at epoch {epoch+1} with Validation Loss: {val_loss}')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val loss: {val_loss/len(val_loader)}, time: {time.time() - start}')

# Entry point for running the training
if __name__ == '__main__':
    train_model(num_epochs=100)
    # print(len(dataset))