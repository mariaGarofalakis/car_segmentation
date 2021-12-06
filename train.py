import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import UNET
import TotalLoss
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    remove_background,
    save_metrics
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/trainset"
TEST_IMG_DIR = "C:/Users/maria/Desktop/project_deep/car_segmentation/testset"
BATCH_SIZE = 6
NUM_EPOCHS = 200

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





def train_fn(loader, model, optimizer, loss_fn, scaler,model2):

    loop = tqdm(loader)

    for batch_idx, all_data in enumerate(loop):

        new_data = remove_background(all_data, model2)
        data = new_data[:, 0, :, :]
        targets = new_data[:, 1:10, :, :]
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

    model2 = UNET(in_channels=1, out_channels=1)
    load_checkpoint(torch.load("remove_background_pretrained_model.pth.tar"), model2)

    model = UNET(in_channels=1, out_channels=9).to(DEVICE)

    loss_fn = TotalLoss.Total_loss()
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
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


 #   check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    test_accuracy = []

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler,model2)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        tmp_metrics = check_accuracy( train_loader ,test_loader, model, device=DEVICE)

        test_accuracy.append(tmp_metrics[2]*100)

        save_metrics(tmp_metrics, 'C:/Users/maria/Desktop/project_deep/car_segmentation/metrics/metrics.csv')

        # print some examples to a folder

 #       save_predictions_as_imgs(
 #           test_loader, model,model2, folder="saved_images/", device=DEVICE
 #       )




if __name__ == '__main__':
    main()

    print('teloas')


