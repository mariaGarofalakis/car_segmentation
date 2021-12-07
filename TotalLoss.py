import torch.nn as nn
import torch
import torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Total_loss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.3, beta=0.7, sigma=0.4 , theta=0.6  ):
        super(Total_loss, self).__init__()
        self.tversky = 0.0
        self.ce = 0.0
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.theta = theta

    def forward(self, inputs, targets, smooth=1e-4):
        #class_weight = torch.tensor([ 0.04, 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12, 0.12, 0.12])
        class_weight = torch.tensor([ 0.5, 1 , 1 , 1 , 1 , 1 , 1, 1, 1])
        crossLoss = nn.CrossEntropyLoss(weight= class_weight.cuda() ,reduction='mean', )
        self.ce = self.sigma *crossLoss(inputs, targets)

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

        self.tversky = self.theta * (1 - (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth))
        Dice_BCE = self.ce + self.tversky

        return Dice_BCE