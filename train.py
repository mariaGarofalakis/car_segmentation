import torch
import torchvision
from tqdm import tqdm
from model import UNET
import TotalLoss
import torch.optim as optim
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    save_metrics,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    check_top_five,
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
LOAD_MODEL = False
alpha = 0.3 #Tversky hyperparameters
beta = 0.7  #Tversky hyperparameters
sigma = 0.4 #Proportion of Cross entropy loss
theta = 0.6 #Proportion of Tversk loss

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

    # If model is on the top 5 save it
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    check_top_five("../checkpoints/baseline/", "baseline.json", loss_fn.tversky.item(), checkpoint)

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
    scaler = torch.cuda.amp.GradScaler()

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

    for epoch in range(NUM_EPOCHS):
        print(f"Traing epoch: {epoch}.............................")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # check accuracy
        tmp_metrics = check_accuracy(train_loader ,test_loader, model, device=DEVICE)
        #Save metrics
        save_metrics(tmp_metrics,'../metrics/baseline.csv')

if __name__ == '__main__':
    main()


