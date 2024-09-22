import easyocr
import torch
import cv2
from student import SimpleCRAFT, mediumCRAFT # Make sure this is the correct import path
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def load_model(model_path):
    device = torch.device("cuda")
    model = mediumCRAFT().to(device)
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

import time
def main(reader, image_path):

    total = 0
    # for i in range(100):
    start = time.time()
    res = reader.readtext(image_path, text_threshold=0)
    t = (time.time() - start)
    total += t
    # print(total/100)
    return res

if __name__ == "__main__":
    model_path = "checkpoints/best.pth"
    dir = "dataset/testing_data/images"
    # image_path = "scan2.jpg"
    # image_path = "dataset/testing_data/images/"

    model, device = load_model(model_path)
    model = model.to(device)
    reader = easyocr.Reader(['en'], gpu=True)
    reader.detector = model
    import os
    filenames = os.listdir(dir)
    for i in range(len(filenames)):
        image_path = os.path.join(dir, filenames[i])
        res = main(reader, image_path)

        image = cv2.imread(image_path)
        # print(res)
        for bbox, word ,score in res:
            tl = bbox[0]
            br = bbox[2]
            tl[0] = int(tl[0])
            tl[1] = int(tl[1])
            br[0] = int(br[0])
            br[1] = int(br[1])
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        # image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print(res)
