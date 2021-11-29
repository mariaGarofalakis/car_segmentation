import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNET
import torch.nn as nn
import ComboLoss
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_imgs_of_car_removing_background,

)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cpu"
TRAIN_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/trainset"
TEST_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/testset"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
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



    train_loader, test_loader = get_loaders(
        TEST_IMG_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )




 #   check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    test_accuracy = []
    test_dice = []
    train_accuracy = []
    train_dice = []
    train_iter = []
    model2 = UNET(in_channels=1, out_channels=1)
    load_checkpoint(torch.load("remove_background_pretrained_model.pth.tar"), model2)

    for epoch in range(NUM_EPOCHS):


        # print some examples to a folder
        save_imgs_of_car_removing_background(
            test_loader,model2, folder="saved_images_after_removing_back/")




if __name__ == '__main__':
    main()

    print('teloas')


