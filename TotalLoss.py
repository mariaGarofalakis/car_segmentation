import torch.nn as nn
import torch
import torch.nn.functional as F

class Total_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Total_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-4, alpha=0.3, beta=0.7 ):
        class_weight = torch.tensor([ 0.5 , 1 , 1 , 1 , 1, 1, 1, 1, 1 ])
        crossLoss = nn.CrossEntropyLoss(weight= class_weight.cuda() ,reduction='mean', )
        CE = crossLoss(inputs, targets)

        inputs = torch.softmax(inputs, 1)
        # flatten label and prediction tensors
        inputs_f = inputs.view(-1)
        targets_f = targets.view(-1)
        inputs = inputs[:, 1:9, :, :].reshape(-1)
        targets = targets[:, 1:9, :, :].reshape(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs* targets).sum()
        FP = ((1 - targets_f) * inputs_f).sum()
        FN = (targets_f * (1 - inputs_f)).sum()

        Tversky = 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        Dice_BCE = 0.4*CE + 0.6*Tversky

        return Dice_BCE