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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path, box, imgH, imgW, PAD=True):
    """Crop and preprocess the image based on the bounding box (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    image = Image.open(image_path).convert('L')
    cropped = image.crop((x1, y1, x2, y2))
    cropped = cropped.resize((imgW, imgH), Image.BICUBIC)
    image_arr = np.array(cropped) / 255.0  # Normalize to [0,1]
    image_arr = np.expand_dims(image_arr, axis=0)  # (C,H,W)
    image_tensor = torch.FloatTensor(image_arr).unsqueeze(0)  # (N,C,H,W)
    return image_tensor.to(device)


def detect_and_recognize_with_display(opt):
    """Detect text regions, recognize text, and place them on a blank image side-by-side with the original."""
    # Initialize EasyOCR Reader
    reader = Reader(['en'], gpu=torch.cuda.is_available())

    # Detect text regions
    detection_results = reader.detect(opt.image_path, link_threshold = 0.9, width_ths=0.5, height_ths=0.5)
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
    with torch.no_grad():
        for idx, box in enumerate(hor_list):
            # EasyOCR returns box as (x1, x2, y1, y2)
            # Reorder to (x1,y1,x2,y2) for cropping
            x1, x2, y1, y2 = box
            box_reordered = [x1, y1, x2, y2]

            image_tensor = preprocess_image(opt.image_path, box_reordered, opt.imgH, opt.imgW, opt.PAD)

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

            recognized_texts.append((pred, (x1, y1, x2, y2)))
            print(f"Recognized Text: {pred}, Confidence Score: {confidence_score:.4f}")

    # Create a blank image of the same size as the original
    original_image = Image.open(opt.image_path).convert('RGB')
    width, height = original_image.size

    # Draw detection bounding boxes on the original image
    draw_orig = ImageDraw.Draw(original_image)
    for box in hor_list:
        x1, x2, y1, y2 = box
        left, top, right, bottom = x1, y1, x2, y2
        draw_orig.rectangle([left, top, right, bottom], outline='red', width=2)

    text_image = Image.new('RGB', (width, height), 'white')
    draw_text = ImageDraw.Draw(text_image)

    # Draw recognized text at the corresponding coordinates on the blank image
    for (text, (x1, y1, x2, y2)) in recognized_texts:
        box_width = x2 - x1
        box_height = y2 - y1

        # Estimate font size from box height
        font_size = 40
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
    combined_image.save('combined_output.png')
    print("Combined image saved as 'combined_output.png'.")
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
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--Transformation', default='TPS', type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', default='ResNet', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', default='Attn', type=str, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')

    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    detect_and_recognize_with_display(opt)
