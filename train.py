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


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = r"C:\Users\aleko\Desktop\segmentation_data\trainset"
TEST_IMG_DIR = r"C:\Users\aleko\Desktop\segmentation_data\testset"
BATCH_SIZE = 6
NUM_EPOCHS = 500

NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True
alpha = 0.3 #Tversky hyperparameters
beta = 0.7  #Tversky hyperparameters
sigma = 0.4 #Proportion of Cross entropy loss
theta = 0.6 #Proportion of Tversk loss



def plot_images(data):

    data = data.numpy().astype(np.uint8)
    image = data[:1,:,:]
    masks = data[1:]

    img = np.transpose(image, (1, 2, 0))
    masks = np.transpose(masks, (1, 2, 0))

    fig, ax = plt.subplots(1, 11, figsize=(10, 3))
    ax[0].imshow(img)

    for it in range(1, len(ax)):
        ax[it].imshow(masks[:,:,it - 1])
    plt.show()


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, all_data in enumerate(loop):
        data = all_data[:, 0, :, :]
        targets = all_data[:, 1:10, :, :]
        data = data.float().unsqueeze(1).to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    print(f"Total Loss: {loss.item()}...Tversky loss: {loss_fn.tversky}... Cross Entropy: {loss_fn.ce}")

def main():
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

    test_transforms  = torchvision.transforms.Compose([
        Normalize(),
        Rescale(256),
        Grayscale(),
        ToTensor(),
    ])

    model = UNET(in_channels=1, out_channels=9).to(DEVICE)
    loss_fn = TotalLoss.Total_loss(alpha=alpha , beta=beta , sigma=sigma , theta=theta)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE , weight_decay=1e-5 , amsgrad=True )

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("final_model.pth.tar"), model)


 #   check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    #lambda1 = lambda epoch: 0.99 ** epoch
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(NUM_EPOCHS):
        print(f"Traing epoch: {epoch}.............................")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,filename="final_model.pth.tar")
        #scheduler.step()

        # check accuracy
        tmp_metrics = []
        tmp_metrics = check_accuracy( train_loader ,test_loader, model, device=DEVICE)

        save_metrics(tmp_metrics,'../metrics/metrics_final.csv')

        # print some examples to a folder
        #save_predictions_as_imgs(
        #    test_loader, model, folder="saved_images/", device=DEVICE
        #)

if __name__ == '__main__':
    main()

    print('teloas')


