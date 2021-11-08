import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from utilis import get_loaders
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = "C:/Users/maria/Desktop/mathimata/deep/project/car_segmentation_2021/clean_data"
BATCH_SIZE = 13
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

def plot_images(data):


    data = data.numpy().astype(np.uint8)
    image = data[:3,:,:]
 #   image = np.expand_dims(image, axis=0)
    masks = data[3:]



    img = np.transpose(image, (1, 2, 0))

    masks = np.transpose(masks, (1, 2, 0))

    fig, ax = plt.subplots(1, 11, figsize=(10, 3))
    ax[0].imshow(img)

    for it in range(1, len(ax)):
        ax[it].imshow(masks[:,:,it - 1])
    plt.show()




def train_fn(loader):

    image = next(iter(loader))
    plot_images(image[0])



if __name__ == '__main__':

    transform = torchvision.transforms.Compose([
   #     torchvision.transforms.Resize(256),
   #     torchvision.transforms.RandomCrop(224),
   #     torchvision.transforms.RandomHorizontalFlip(),
        Normalize(),
        Rescale(256),
     #   Grayscale(),
        randomHorizontalFlip(),
        randomShiftScaleRotate(),
        randomHueSaturationValue(),
        randomZoom(),
        ToTensor(),
            ])

    train_loader = get_loaders(train_dir=TRAIN_IMG_DIR, batch_size=BATCH_SIZE, train_transform=transform, num_workers=NUM_WORKERS,
                               pin_memory=PIN_MEMORY)
    train_fn(train_loader)



    print('teloas')


