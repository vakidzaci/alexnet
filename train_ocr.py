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
# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, filenames, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

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

train_dir = 'dataset/training_data/images'
val_dir = 'dataset/training_data/images'

train_filenames = os.listdir(train_dir)
val_filenames = os.listdir(val_dir)

# Load your dataset
train_dataset = CustomDataset(train_dir, train_filenames, transform=data_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = CustomDataset(val_dir, val_filenames, transform=data_transforms)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True, num_workers=4)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model = SimpleCRAFT().to(device)
student_model.train()
print(sum(dict((p.data_ptr(), p.numel()) for p in student_model.parameters()).values()))
teacher_model = CRAFT().to(device)
teacher_model.load_state_dict(copyStateDict(torch.load("/home/vakidzaci/.EasyOCR/model/craft_mlt_25k.pth", map_location=device)))
teacher_model.eval()

# Loss function and optimizer
criterion = nn.MSELoss()
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
                teacher_outputs = teacher_model(images)

            student_outputs = student_model(images)
            loss = criterion(student_outputs[0], teacher_outputs[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            # Validation loss
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                teacher_outputs = teacher_model(images)
                student_outputs = student_model(images)
                loss = criterion(student_outputs[0], teacher_outputs[0])
                # print(teacher_outputs[0].size(), student_outputs[0].size())
                val_loss += loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), f'checkpoints/best_model_epoch_{epoch+1}_val_loss_{val_loss}.pth')
            print(f'Saved new best model at epoch {epoch+1} with Validation Loss: {val_loss}')

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val loss: {val_loss/len(val_loader)}, time: {time.time() - start}')

# Entry point for running the training
if __name__ == '__main__':
    train_model(num_epochs=100)
