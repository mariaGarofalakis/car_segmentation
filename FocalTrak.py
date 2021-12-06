import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        BCE = loss_fn(inputs, targets, )

        inputs = F.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)


        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP +  1e-4) / (TP + 0.3 * FP + 0.7 * FN +  1e-4)
        FocalTversky = (1 - Tversky) ** 1

        total = 0.4 * BCE + 0.6 * FocalTversky

        return total



