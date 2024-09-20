import torch
import time
import numpy as np
from student import SimpleCRAFT  # Import your student model
from easyocr.craft import CRAFT  # Import the teacher model (pre-trained CRAFT model)
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from collections import OrderedDict

# Create a random 1000x1000 RGB image
image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255  # Convert to PyTorch tensor and normalize

# Define transformation (resize if necessary and normalize as per model requirements)
transform = Compose([
    Resize((256, 256)),  # Resize the image if required by the model
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformations
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension
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
# Initialize models
device = torch.device("cpu")  # Ensure models and data are on CPU
student_model = SimpleCRAFT().to(device)
teacher_model = CRAFT().to(device)
teacher_model.load_state_dict(copyStateDict(torch.load("/home/vakidzaci/.EasyOCR/model/craft_mlt_25k.pth", map_location=device)))
teacher_model.eval()

# Ensure models are in evaluation mode
student_model.eval()
teacher_model.eval()

# Function to measure inference time
def measure_inference_time(model, input_image, runs=100):
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_image)

    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_image)
    end_time = time.time()
    return (end_time - start_time) / runs  # Average time per inference

# Measure inference time
student_time = measure_inference_time(student_model, image, 100)
teacher_time = measure_inference_time(teacher_model, image, 100)

print(f"Average inference time on CPU - Student Model: {student_time:.5f} seconds")
print(f"Average inference time on CPU - Teacher Model: {teacher_time:.5f} seconds")
