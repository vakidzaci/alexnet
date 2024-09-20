import torch
import cv2
from student import SimpleCRAFT  # Make sure this is the correct import path
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCRAFT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Assuming model expects 256x256 RGB images
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def draw_boxes(image, boxes):
    for box in boxes:
        top_left, bottom_right = box[0], box[1]
        cv2.rectangle(image, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(model_path, image_path):
    model, device = load_model(model_path)
    input_image = preprocess_image(image_path)
    input_image = input_image.to(device)

    with torch.no_grad():
        boxes = model(input_image)  # Ensure model outputs bounding box coordinates in [(x1, y1, x2, y2), ...] format

    original_image = cv2.imread(image_path)  # Load original image again to draw boxes
    draw_boxes(original_image, boxes.cpu().numpy())  # Convert boxes to NumPy array if necessary


if __name__ == "__main__":
    model_path = "path_to_your_model.pth"
    image_path = "path_to_your_test_image.jpg"
    main(model_path, image_path)
