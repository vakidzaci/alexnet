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

import cv2
import numpy as np

def segment_sentence_into_words(sentence_image):
    """
    Segments a sentence image into individual word images.

    Args:
        sentence_image (numpy.ndarray): Grayscale or binary image of the sentence.

    Returns:
        List[numpy.ndarray]: List of cropped word images.
    """
    # Ensure the image is binary
    if len(sentence_image.shape) == 3:  # If the image is not grayscale
        sentence_image = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(sentence_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Compute the vertical projection profile
    vertical_projection = np.sum(dilated_image, axis=0)

    # Identify the boundaries where the projection is zero (valleys)
    threshold = 0  # Adjust if necessary for noisy images
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

    # Handle the last word if the image ends with text
    if in_word:
        word_boundaries.append((start_idx, len(vertical_projection)))

    # Crop word images based on identified boundaries
    word_images = []
    for start, end in word_boundaries:
        if end - start > 10:  # Filter out very small segments
            word_image = sentence_image[:, start:end]
            word_images.append(word_image)

    return word_images


def preprocess_image(cropped, imgH, imgW, PAD=True, binary=False):
    """Crop and preprocess the image based on the bounding box (x1,y1,x2,y2)."""

    cropped = cropped.resize((imgW, imgH), Image.BICUBIC)
    image_arr = np.array(cropped)

    if binary:
        image_arr = cv2.adaptiveThreshold(image_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    # print(image_arr)
    image_arr = image_arr / 255.0  # Normalize to [0,1]
    image_arr = np.expand_dims(image_arr, axis=0)  # (C,H,W)
    image_tensor = torch.FloatTensor(image_arr).unsqueeze(0)  # (N,C,H,W)

    return image_tensor.to(device)


def predict(image_tensor, model, converter):

    length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
    preds = model(image_tensor, text_for_pred, is_train=False)

    # Decode predictions
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index, length_for_pred)

    # Confidence score
    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    pred = preds_str[0]
    pred_EOS = pred.find('[s]')
    pred = pred[:pred_EOS]
    pred_max_prob = preds_max_prob[0, :pred_EOS]
    confidence_score = pred_max_prob.cumprod(dim=0)[-1] if pred_max_prob.nelement() > 0 else 0.0
    return pred, confidence_score

def detect_and_recognize_with_display(opt):
    """Detect text regions, recognize text, and place them on a blank image side-by-side with the original."""
    # Initialize EasyOCR Reader
    reader = Reader(['en'], gpu=torch.cuda.is_available(), recognizer=False)

    # Detect text regions
    detection_results = reader.detect(opt.image_path)
    if not detection_results:
        print("No text regions detected.")
        return

    hor_list, free_list = detection_results
    hor_list, free_list = hor_list[0], free_list[0]

    # Initialize the recognition model
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

    # Recognize text from each detected region
    score_color = []
    confidences = []
    confidences_by_word_count = {}
    image = Image.open(opt.image_path).convert('L')
    original_image = Image.open(opt.image_path).convert('RGB')
    width, height = original_image.size
    draw_orig = ImageDraw.Draw(original_image)

    with torch.no_grad():
        for idx, box in enumerate(hor_list):
            # EasyOCR returns box as (x1, x2, y1, y2)
            # Reorder to (x1,y1,x2,y2) for cropping
            x1, x2, y1, y2 = box
            cropped = image.crop((x1, y1, x2, y2))

            image_tensor = preprocess_image(cropped, opt.imgH, opt.imgW, opt.PAD)
            pred, confidence_score = predict(image_tensor, model, converter)

            # if len(pred.split(" ")) == 1:

            # left, top, right, bottom = x1, y1, x2, y2
            # draw_orig.rectangle([left, top, right, bottom], outline='red', width=2)

            recognized_texts.append((pred, (x1, y1, x2, y2)))
                # continue
            # else:
            #     texts = []
            #     confidence_scores = []
            #     width_word = cropped.width
            #     width_char = int(width_word/len(pred))
            #     idxs = []
            #     for uuuu, c in enumerate(pred):
            #         if c == " ":
            #             idxs.append((uuuu+0.5)*width_char)
            #
            #     x_points = [0] + idxs + [width_word]
            #     for jjjj in range(0, len(x_points) - 2, 2):
            #         left = x_points[jjjj]
            #         right = x_points[jjjj + 2]
            #         draw_orig.rectangle([x1 + left, y1, x1 + right, y2], outline='green', width=2)
            #
            #         segment = image.crop((x1 + left, y1, x1 + right, y2))
            #         # draw_orig.rectangle([left, top, right, bottom], outline='red', width=2)
            #
            #         # segment = image.crop((x1 + left, y1, x2 + right, y2))
            #         # segment = cropped.crop((left, 0, right, image.height))
            #         image_tensor = preprocess_image(segment, opt.imgH, opt.imgW, opt.PAD)
            #
            #         pred, confidence_score = predict(image_tensor, model, converter)
            #         texts.append(pred)
            #         confidence_scores.append(float(confidence_score))

                # pred = " ".join(texts)
                # recognized_texts.append((pred, (x1, y1, x2, y2)))
                # confidence_score = sum(confidence_scores)/len(confidence_scores)


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



    # Create a blank image of the same size as the original


    # Draw detection bounding boxes on the original image

    for i, box in enumerate(hor_list):
        x1, x2, y1, y2 = box
        left, top, right, bottom = x1, y1, x2, y2
        # draw_orig.rectangle([left, top, right, bottom], outline='red', width=2)
        draw_orig.rectangle([left, top, right, bottom], outline=score_color[i], width=2)

    text_image = Image.new('RGB', (width, height), 'white')
    draw_text = ImageDraw.Draw(text_image)

    # Draw recognized text at the corresponding coordinates on the blank image
    for (text, (x1, y1, x2, y2)) in recognized_texts:
        box_width = x2 - x1
        box_height = y2 - y1

        # Estimate font size from box height
        font_size = 35
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default(font_size)

        # Adjust font size if needed
        while True:
            x0, y0, x1_text, y1_text = font.getbbox(text)
            text_width = x1_text - x0
            text_height = y1_text - y0
            if text_width <= box_width and text_height <= box_height:
                break
            font_size -= 1
            if font_size < 8:
                # Too small to shrink further
                break
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default(font_size)

        # Center text within the bounding box
        x0, y0, x1_text, y1_text = font.getbbox(text)
        text_height = y1_text - y0
        text_width = x1_text - x0
        text_x = x1 + (box_width - text_width) // 2
        text_y = y1 + (box_height - text_height) // 2

        draw_text.text((text_x, text_y), text, font=font, fill='black')

    # Combine original with bounding boxes and text image side-by-side
    combined_image = Image.new('RGB', (2 * width, height), 'white')
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(text_image, (width, 0))

    # Save and show the combined image
    combined_image.save('combined_output2.png')
    print("Combined image saved as 'combined_output.png'.")
    print(f"confidence average: {sum(confidences)/len(confidences)}")
    print(confidences_by_word_count)
    keys = list(confidences_by_word_count.keys())
    keys.sort()
    for key in keys:
        val = confidences_by_word_count[key]
        print(key, sum(val)/len(val))
    combined_image.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')

    parser.add_argument('--image_path', required=True, help='path to the image file')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=200, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--character', type=str, default=" «»!%&'()*+,-./0123456789:;<=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]_abcdefghijklmnopqrstuvwxyz|ІАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёіҒғҚқҢңҮүҰұӘәӨө", help='character label')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--Transformation', default='TPS', type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', default='ResNet', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', default='Attn', type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=256,
                        help='the number of output channel of Feature extractor')

    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    detect_and_recognize_with_display(opt)
