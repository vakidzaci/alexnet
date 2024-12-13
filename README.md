import string
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from utils import AttnLabelConverter
from model import Model
from easyocr import Reader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import OrderedDict
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def segment_sentence_into_words(sentence_image):
    """
    Segments a sentence image into individual word images.
    """
    if len(sentence_image.shape) == 3:  # If the image is not grayscale
        sentence_image = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(sentence_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    vertical_projection = np.sum(dilated_image, axis=0)

    threshold = 0
    word_boundaries = []
    in_word = False
    start_idx = 0

    for i, value in enumerate(vertical_projection):
        if value > threshold and not in_word:
            start_idx = i
            in_word = True
        elif value <= threshold and in_word:
            word_boundaries.append((start_idx, i))
            in_word = False

    if in_word:
        word_boundaries.append((start_idx, len(vertical_projection)))

    word_images = []
    for start, end in word_boundaries:
        if end - start > 10:
            word_image = sentence_image[:, start:end]
            word_images.append(word_image)

    return word_images


def preprocess_image(cropped, imgH, imgW, PAD=True, binary=False):
    """Crop and preprocess the image."""
    cropped = cropped.resize((imgW, imgH), Image.BICUBIC)
    image_arr = np.array(cropped)

    if binary:
        image_arr = cv2.adaptiveThreshold(image_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                          cv2.THRESH_BINARY, 11, 2)

    image_arr = image_arr / 255.0  # Normalize to [0,1]
    image_arr = np.expand_dims(image_arr, axis=0)  # (C,H,W)
    image_tensor = torch.FloatTensor(image_arr).unsqueeze(0)  # (N,C,H,W)

    return image_tensor.to(device)


def predict(image_tensor, model, converter, opt):
    """
    Run prediction on a batch of images.
    image_tensor: (batch_size, C, H, W)
    """
    batch_size = image_tensor.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    preds = model(image_tensor, text_for_pred, is_train=False)

    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index, length_for_pred)
    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)

    results = []
    for i, pred in enumerate(preds_str):
        pred_EOS = pred.find('[s]')
        pred_text = pred[:pred_EOS]
        pred_max_prob = preds_max_prob[i, :pred_EOS]
        confidence_score = pred_max_prob.cumprod(dim=0)[-1].item() if pred_max_prob.nelement() > 0 else 0.0
        results.append((pred_text, confidence_score))
    return results


def detect_and_recognize_with_display(opt):
    reader = Reader(['en'], gpu=torch.cuda.is_available(), recognizer=False)

    detection_results = reader.detect(opt.image_path)
    if not detection_results:
        print("No text regions detected.")
        return

    hor_list, free_list = detection_results
    hor_list, free_list = hor_list[0], free_list[0]

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt)
    state_dict = torch.load(opt.saved_model, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    print('Loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    recognized_texts = []
    score_color = []
    confidences = []
    confidences_by_word_count = {}

    image = Image.open(opt.image_path).convert('L')
    original_image = Image.open(opt.image_path).convert('RGB')
    width, height = original_image.size
    draw_orig = ImageDraw.Draw(original_image)

    # Preprocess all images and stack them into a batch
    image_tensors = []
    boxes = []
    for idx, box in enumerate(hor_list):
        x1, x2, y1, y2 = box
        cropped = image.crop((x1, y1, x2, y2))
        image_tensor = preprocess_image(cropped, opt.imgH, opt.imgW, opt.PAD)
        image_tensors.append(image_tensor)
        boxes.append(box)

    # Combine all tensors into one batch
    if len(image_tensors) == 0:
        print("No detected text regions to process.")
        return
    batch_image_tensor = torch.cat(image_tensors, dim=0)

    # Predict for the entire batch
    with torch.no_grad():
        predictions = predict(batch_image_tensor, model, converter, opt)

    # Process predictions
    for i, (pred, confidence_score) in enumerate(predictions):
        x1, x2, y1, y2 = boxes[i]
        recognized_texts.append((pred, (x1, y1, x2, y2)))

        print(f"Recognized Text: {pred}, Confidence Score: {confidence_score:.4f}")

        if len(pred) not in confidences_by_word_count:
            confidences_by_word_count[len(pred)] = [float(confidence_score)]
        else:
            confidences_by_word_count[len(pred)].append(float(confidence_score))

        confidences.append(confidence_score)

        if confidence_score > 0.9:
            score_color.append('green')
        elif confidence_score > 0.5:
            score_color.append('blue')
        else:
            score_color.append('red')

    # Draw bounding boxes
    for i, box in enumerate(hor_list):
        x1, x2, y1, y2 = box
        left, top, right, bottom = x1, y1, x2, y2
        draw_orig.rectangle([left, top, right, bottom], outline=score_color[i], width=2)

    text_image = Image.new('RGB', (width, height), 'white')
    draw_text = ImageDraw.Draw(text_image)

    for (text, (x1, y1, x2, y2)) in recognized_texts:
        box_width = x2 - x1
        box_height = y2 - y1

        font_size = 35
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Adjust font size
        while True:
            x0, y0, x1_text, y1_text = font.getbbox(text)
            text_width = x1_text - x0
            text_height = y1_text - y0
            if text_width <= box_width and text_height <= box_height:
                break
            font_size -= 1
            if font_size < 8:
                break
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        # Center text
        x0, y0, x1_text, y1_text = font.getbbox(text)
        text_height = y1_text - y0
        text_width = x1_text - x0
        text_x = x1 + (box_width - text_width) // 2
        text_y = y1 + (box_height - text_height) // 2

        draw_text.text((text_x, text_y), text, font=font, fill='black')

    combined_image = Image.new('RGB', (2 * width, height), 'white')
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(text_image, (width, 0))

    combined_image.save('combined_output2.png')
    print("Combined image saved as 'combined_output2.png'.")
    print(f"confidence average: {sum(confidences)/len(confidences) if confidences else 0}")
    print(confidences_by_word_count)
    keys = list(confidences_by_word_count.keys())
    keys.sort()
    for key in keys:
        val = confidences_by_word_count[key]
        print(key, sum(val)/len(val))
    combined_image.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--sensitive', action='store_true')

    parser.add_argument('--image_path', required=True)
    parser.add_argument('--saved_model', required=True)
    parser.add_argument('--batch_max_length', type=int, default=200)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--PAD', action='store_true')
    parser.add_argument('--character', type=str, default=" «»!%&'()*+,-./0123456789:;<=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_abcdefghijklmnopqrstuvwxyz|ІАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіҒғҚқҢңҮүҰұӘәӨө")
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--Transformation', default='TPS', type=str)
    parser.add_argument('--FeatureExtraction', default='ResNet', type=str)
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str)
    parser.add_argument('--Prediction', default='Attn', type=str)
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=256)
    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    detect_and_recognize_with_display(opt)
