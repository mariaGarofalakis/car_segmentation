import torch
import torchvision
from tqdm import tqdm
from model import UNET
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import numpy as np
import IoULoss
from transforms import Rescale, Normalize, ToTensor, randomHueSaturationValue, randomHorizontalFlip, randomZoom, Grayscale, randomShiftScaleRotate
from utilis import (
    load_checkpoint,
    save_checkpoint_background,
    get_loaders,
    check_accuracy_background,
)


# Hyperparameters etc.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

class optuna_train_oneClass(object):
    def __init__(self):
        self.train_image_dir = "C:/Users/maria/Desktop/project_deep/car_segmentation/trainset"
        self.test_image_dir = "C:/Users/maria/Desktop/project_deep/car_segmentation/testset"
        self.IoU = 1000

    def suggest_hyperparameters(self, trial):
        # Learning rate on a logarithmic scale
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        # Batch size  in the range from 2 to 6 with step size 1
        batch_size = int(trial.suggest_float("batch_size", 2, 6, step=1))
        optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])

        return lr, batch_size, optimizer_name




    def train(self, trial):
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

        lr, batch_size, optimizer_name = self.suggest_hyperparameters(trial)

        # Pick an optimizer based on Optuna's parameter suggestion
        params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr,
                                        momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        loss_fn =IoULoss.IoULoss()


        train_loader, test_loader = get_loaders(
            self.train_image_dir,
            self.test_image_dir,
            batch_size,
            train_transform,
            test_transforms,
            NUM_WORKERS,
            PIN_MEMORY,
        )

        if LOAD_MODEL:
            load_checkpoint(torch.load("remove_background_pretrained_model.pth.tar"), model)


     #   check_accuracy(test_loader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        IoU = []
        for epoch in range(NUM_EPOCHS):
            ############################################ train_fn ####################################################

            loop = tqdm(train_loader)

            for batch_idx, all_data in enumerate(loop):
                data = all_data[:, 0, :, :]
                targets = all_data[:, 10, :, :]
                data = data.float().unsqueeze(1).to(device=DEVICE)
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
            ################################################################################################################
            lr_scheduler.step()
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint_background(checkpoint)

            # check accuracy
            test_IoU =check_accuracy_background(train_loader,test_loader, model, device=DEVICE)[1][2]
            IoU.append(test_IoU)

            trial.report(test_IoU, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        optuna_IoU = sum(IoU) / len(IoU)
        if (optuna_IoU <= self.IoU):
            self.IoU = optuna_IoU
            print(self.IoU)

        return optuna_IoU



if __name__ == '__main__':
    trainObj = optuna_train_oneClass()
    study = optuna.create_study(study_name="Final-project-optuna", direction="minimize",
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=5, n_warmup_steps=1
                                ))
    study.optimize(trainObj.train, n_trials=15)

    # Initialize the best_val_loss value
    # mean_AP_accuracy = best_val_loss = float('Inf')

    # if mean_AP_accuracy <= best_val_loss:
    #     best_val_loss = mean_AP_accuracy

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    lr = 0
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe()
    df.to_csv(r'optuna_metrics.csv', index=False)
    fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('fig_history.png')
    fig_param_importances = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('fig_param_importances.png')
    plot_edf = optuna.visualization.matplotlib.plot_edf(study)
    plt.savefig('plot_edf.png')
    intermediate_values = optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig('intermediate_values.png')
    parallel_coordinate = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig('parallel_coordinate.png')

    print('teloas')


