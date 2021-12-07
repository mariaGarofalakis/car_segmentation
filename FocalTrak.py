import torch.nn as nn
import torch
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = 0.0
        self.BCE = 0.0

    def forward(self, inputs, targets, smooth=1e-4, alpha=0.3, beta=0.7, sigma=0.4 , theta=0.6 ):

        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        self.BCE = sigma*loss_fn(inputs, targets, )

        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        self.tversky = theta * (1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth))

        total = self.BCE + self.tversky

        return total



