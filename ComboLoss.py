import torch.nn as nn
import torch
import torch.nn.functional as F

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        ALPHA = 0.8  # < 0.5 penalises FP more, > 0.5 penalises FN more
        CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss
        eps = 1e-9
        # comment out if your model contains a sigmoid or equivalent activation layer

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs, 1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * (1-IoU))

        return combo



