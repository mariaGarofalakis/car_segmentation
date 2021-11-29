import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNET
import torch.nn as nn
import IoULoss
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    load_checkpoint,
    save_checkpoint_background,
    get_loaders,
    check_accuracy_background,
    save_imgs_of_car_removing_background,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/trainset"
TEST_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/testset"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True





def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, all_data in enumerate(loop):
        data = all_data[:, 0, :, :]
        targets = all_data[:, 10, :, :]
        data = data.float().unsqueeze(1).to(device=DEVICE)
    #    targets = targets.float().to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

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

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
 #   loss_fn = nn.CrossEntropyLoss() #softDice   weighted average of both
    loss_fn =IoULoss.IoULoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("remove_background_pretrained_model.pth.tar"), model)


 #   check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()



    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint_background(checkpoint)

        # check accuracy
        check_accuracy_background(test_loader, model, device=DEVICE)

        # print some examples to a folder
        save_imgs_of_car_removing_background(
            test_loader, model, folder="saved_no_back_images/", device=DEVICE
        )


if __name__ == '__main__':
    main()

    print('teloas')


