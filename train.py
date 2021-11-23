import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNET
import torch.nn as nn
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = r'C:\Users\aleko\Desktop\segmentation_data\clean_data\clean_data'
BATCH_SIZE = 6
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True



def plot_images(data):


    data = data.numpy().astype(np.uint8)
    image = data[:1,:,:]
 #   image = np.expand_dims(image, axis=0)
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
    #    targets = targets.float().unsqueeze(1).to(device=DEVICE)

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
        #     torchvision.transforms.Resize(256),
        #     torchvision.transforms.RandomCrop(224),
        #     torchvision.transforms.RandomHorizontalFlip(),
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


 #   check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    test_accuracy = []
    test_dice = []
    train_accuracy = []
    train_dice = []
    train_iter = []

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        train_tmp = 0.0
        train_tmp_dc = 0.0
        test_tmp = 0.0
        test_tmp_dc = 0.0

        # check accuracy
        train_tmp,train_tmp_dc,test_tmp,test_tmp_dc = check_accuracy( train_loader ,test_loader, model, device=DEVICE)
        train_accuracy.append(train_tmp*100)
        train_dice.append(train_tmp_dc)
        test_accuracy.append(test_tmp*100)
        test_dice.append(test_tmp_dc)
        train_iter.append(epoch)

        # print some examples to a folder
        save_predictions_as_imgs(
            test_loader, model, folder="saved_images/", device=DEVICE
        )

        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_iter, train_accuracy, label='train_loss')
        plt.plot(train_iter, test_accuracy, label='valid_loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_iter, train_dice, label='train_accs')
        plt.plot(train_iter, test_dice, label='valid_accs')
        plt.legend()
        plt.savefig('metrics.png')

if __name__ == '__main__':
    main()

    print('teloas')


