import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNET
import torch.nn as nn
import TotalLoss
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
import csv
from utilis import (
    save_metrics,
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd

# CONFIG
#
BETA = 2
RETURN_CMATRIX = True
INVALID_ZERO_DIVISON = False
VALID_ZERO_DIVISON = 1.0


#
# METHODS
#
def confusion_matrix(target, prediction, value, ignore_value=None):
    true = (target == prediction)
    false = (~true)
    pos = (target == value)
    neg = (~pos)
    keep = (target != ignore_value)
    tp = (true * pos).sum()
    fp = (false * pos * keep).sum()
    fn = (false * neg * keep).sum()
    tn = (true * neg).sum()
    return _get_items(tp, fp, fn, tn)


def precision(tp, fp, fn):
    return _precision_recall(tp, fp, fn)


def recall(tp, fn, fp):
    return _precision_recall(tp, fn, fp)


def fbeta(p, r, beta=BETA):
    if p is None: p = precision(tp, fp)
    if r is None: r = recall(tp, fn)
    beta_sq = beta ** 2
    numerator = (beta_sq * p + r)
    if numerator:
        return (1 + beta_sq) * (p * r) / numerator
    else:
        return 0


def stats(
        target,
        prediction,
        value,
        ignore_value=None,
        beta=BETA,
        return_cmatrix=RETURN_CMATRIX):
    tp, fp, fn, tn = confusion_matrix(
        target,
        prediction,
        value,
        ignore_value=ignore_value)
    p = precision(tp, fp, fn)
    r = recall(tp, fn, fp)
    stat_values = [p, r]
    if not _is_false(beta):
        stat_values.append(fbeta(p, r, beta=beta))
    if return_cmatrix:
        stat_values += [tp, fp, fn, tn]
    return stat_values


#
# INTERNAL
#
def _precision_recall(a, b, c):
    if (a + b):
        return a / (a + b)
    else:
        if c:
            return INVALID_ZERO_DIVISON
        else:
            return VALID_ZERO_DIVISON


def _is_false(value):
    return value in [False, None]


def _get_items(*args):
    try:
        return list(map(lambda s: s.item(), args))
    except:
        return args

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cpu"

TRAIN_IMG_DIR = r"C:\Users\aleko\Desktop\segmentation_data\trainset"
TEST_IMG_DIR = r"C:\Users\aleko\Desktop\segmentation_data\testset"
BATCH_SIZE = 30
NUM_EPOCHS = 500

NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
alpha = 0.3 #Tversky hyperparameters
beta = 0.7  #Tversky hyperparameters
sigma = 0.4 #Proportion of Cross entropy loss
theta = 0.6 #Proportion of Tversk loss

if __name__ == '__main__':
    train_transform = torchvision.transforms.Compose([
        Normalize(),
        Rescale(256),
        randomHorizontalFlip(),
        randomShiftScaleRotate(),
        randomHueSaturationValue(),
        randomZoom(),
        Grayscale(),
        ToTensor(),
    ])

    test_transforms = torchvision.transforms.Compose([
        Normalize(),
        Rescale(256),
        Grayscale(),
        ToTensor(),
    ])

    model = UNET(in_channels=1, out_channels=9).to(DEVICE)
    loss_fn = TotalLoss.Total_loss(alpha=alpha, beta=beta, sigma=sigma, theta=theta)

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model.eval()
    load_checkpoint(torch.load("../checkpoints/baseline/checkpoint_3.pth.tar"), model)
    print('Loaded')
    cm = np.empty([9, 9])
    with torch.no_grad():
        for idx, all_data in enumerate(test_loader):
            print(idx)
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)

            preds = torch.softmax(model(x), 1)

            for itr in range(9):
                torchvision.utils.save_image(
                    torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)[:, itr, :, :].unsqueeze(
                        1).cuda(), f"{folder}/pred_{idx}_itr_{itr}.png"
                )
                torchvision.utils.save_image(preds[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_grey_{itr}.png")


            # Confusion matrix
            preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)
            for i in range(9):
                y[:,i,:,:] *= (i+1)
                preds[:,i,:,:] *= (i+1)
            preds = torch.sum(preds, 1)
            preds = (preds).view(-1)
            y = torch.sum(y,1)
            y = (y).reshape(-1)
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(y_true=y.numpy(),y_pred=preds.numpy())
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(20, 35))
            labels = ["Background", "Front_door", "Back_door", "Fender", "Frame", "Front_bumber", "Hood", "Back_bumber","Trunk"]
            p = sns.heatmap(cmn, annot=True, fmt='.2f',xticklabels=labels,yticklabels=labels)
            p.set_yticklabels(p.get_yticklabels(), rotation=0)
            p.set_xlabel("Actual",fontsize=20)
            p.xaxis.set_label_position('top')
            p.set_ylabel("Prediction",fontsize=20)
            plt.show()

            #Recall
            #TODO
            #Precision
            #TODO
            #F1 score
            #TODO
            #IoU

