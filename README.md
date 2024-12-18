import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ConvNet(x)


class AdaptiveAvgPoolDynamic(nn.Module):
    """
    This custom layer will adaptively pool only the width dimension down to 1,
    leaving the height dimension unchanged.
    """

    def __init__(self, output_size=(None, 1)):
        super(AdaptiveAvgPoolDynamic, self).__init__()
        if output_size != (None, 1):
            raise ValueError("This custom layer only supports output_size=(None, 1).")

    def forward(self, x):
        # x: [B, C, H, W]
        # We want to keep H the same and reduce W to 1 via average pooling
        B, C, H, W = x.size()
        # Use adaptive_avg_pool2d to pool along the width dimension
        # Set the target output size to (H, 1)
        return F.adaptive_avg_pool2d(x, (H, 1))


# Construct the full model
model = nn.Sequential(
    VGG_FeatureExtractor(),
    AdaptiveAvgPoolDynamic((None, 1))
)




import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

####################################
# Dummy function for contrast adjustment (if needed)

####################################
def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

####################################
# Dataset: Loads images from a list of file paths
####################################
class ListDataset(torch.utils.data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, 'L')


####################################
# Normalization and Padding
####################################
class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        # Convert and normalize: (img - 0.5)/0.5 shifts [0,1] range to [-1,1]
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


####################################
# Collate function for DataLoader
####################################
class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast=0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size

            # Optional contrast adjustment
            if self.adjust_contrast > 0:
                image = np.array(image)
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, 'L')

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


####################################
# Load the feature extractor model
####################################
# This assumes you have saved a truncated model state_dict as `feature_extractor.pth`
# The truncated model should contain something like:
# feature_extractor = nn.Sequential(
#     base_model.FeatureExtraction,
#     base_model.AdaptiveAvgPool
# )
from feature_ext import VGG_FeatureExtractor, AdaptiveAvgPoolDynamic

model = nn.Sequential(
    VGG_FeatureExtractor(),
    AdaptiveAvgPoolDynamic((None, 1))
)
# Instantiate your truncated feature extractor and load weights
feature_extractor = model
feature_extractor.load_state_dict(torch.load("feature_extractor.pth", map_location='cpu'))
feature_extractor.eval()

def calculate_ratio(width,height):
    '''
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = 1./ratio
    return ratio

import cv2
def compute_ratio_and_resize(img,width,height,model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = calculate_ratio(width,height)
        img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.Resampling.LANCZOS)
    else:
        img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.Resampling.LANCZOS)
    return img, ratio


img_list = []

for filename in os.listdir("test_data_for_clustering"):
    img_path = os.path.join("test_data_for_clustering", filename)
    img = cv2.imread(img_path,0)
    w,h = img.shape
    img,_ = compute_ratio_and_resize(img,w,h,32)
    img_list.append(img)

batch_size = 16
workers = 4
imgH = 32
imgW = 100

AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True, adjust_contrast=0.0)
test_data = ListDataset(img_list)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False,
    num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True
)


all_features = []
with torch.no_grad():
    for batch_images in test_loader:
        # batch_images: [B, 1, imgH, imgW]
        features = feature_extractor(batch_images)  # shape depends on your model architecture
        # Suppose features are [B, C, H, W]
        # After AdaptiveAvgPool, H might be 1, so features: [B, C, W]
        # If height dimension is still there, try squeezing: features = features.squeeze(3) if features.dim()==4
        if features.dim() == 4:
            # Expected shape: [B, C, H, W] -> after AdaptiveAvgPool, H should be 1:
            features = features.squeeze(3)  # Now features: [B, C, W]

        # Now pool across width to get a single vector per image
        # Mean pooling across width dimension:
        features_vector = features.mean(dim=2)  # [B, C]

        all_features.append(features_vector.cpu().numpy())

# Concatenate all features
all_features = np.concatenate(all_features, axis=0)  # [N, C] N=total images, C=256 or whatever your channel dim is.

# Save features for future clustering
np.save("image_features.npy", all_features)
print("Features saved to image_features.npy")
