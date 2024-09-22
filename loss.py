import torch.nn as nn
from torch import sigmoid
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1.0
        y_pred = sigmoid(y_pred)  # Apply sigmoid if logits are passed
        intersection = (y_pred * y_true).sum()
        return 1 - (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

# Combine Dice Loss with BCEWithLogitsLoss
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # For raw logits
        self.dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        # Separate region and affinity predictions
        region_pred, affinity_pred = y_pred[:, :, :, 0], y_pred[:, :, :, 1]
        region_true, affinity_true = y_true[:, :, :, 0], y_true[:, :, :, 1]

        # Compute Binary Cross-Entropy Loss
        region_loss_bce = self.bce_loss(region_pred, region_true)
        affinity_loss_bce = self.bce_loss(affinity_pred, affinity_true)

        # Compute Dice Loss
        region_loss_dice = self.dice_loss(region_pred, region_true)
        affinity_loss_dice = self.dice_loss(affinity_pred, affinity_true)

        # Combine BCE and Dice Loss
        total_region_loss = region_loss_bce + region_loss_dice
        total_affinity_loss = affinity_loss_bce + affinity_loss_dice

        total_loss = total_region_loss + total_affinity_loss
        return total_loss
