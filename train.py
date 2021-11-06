import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from utilis import get_loaders
from PIL import Image, ImageFilter
from matplotlib import cm

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

def imshow(img):


    w, h = 512, 512


    npimg = img.numpy()
    skata = np.transpose(npimg, (1, 2, 0))

    im = Image.fromarray(np.uint8(cm.gist_earth(skata) * 255))

    im.save()




def train_fn(loader):

    data = next(iter(loader))
    imshow(data[0])



if __name__ == '__main__':

    transform = torchvision.transforms.Compose([
   #     torchvision.transforms.Resize(256),
   #     torchvision.transforms.RandomCrop(224),
   #     torchvision.transforms.RandomHorizontalFlip(),
      #  torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_loader = get_loaders(train_dir=TRAIN_IMG_DIR, batch_size=BATCH_SIZE, train_transform=transform, num_workers=NUM_WORKERS,
                               pin_memory=PIN_MEMORY)
    train_fn(train_loader)



    print('teloas')


