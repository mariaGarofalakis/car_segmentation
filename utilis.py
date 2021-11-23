import torch
import torchvision
from dataset import create_dataset
from torch.utils.data import DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, batch_size, train_transform, test_transform, num_workers=4, pin_memory=True):
    train_ds_original = create_dataset(image_dir=train_dir, train=True, transform = test_transform)
    train_ds = create_dataset(image_dir=train_dir, train=True, transform = train_transform)

    con_Dataset = torch.utils.data.ConcatDataset([train_ds_original, train_ds])
    train_loader = DataLoader(con_Dataset, batch_size=batch_size,num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)


    test_ds = create_dataset(
        image_dir=train_dir, train=False, transform = test_transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader

def check_accuracy( train_loader ,test_loader, model, device="cuda"):
    num_correct_train = 0
    num_pixels_train = 0
    num_correct_test = 0
    num_pixels_test = 0
    dice_score_train = 0
    dice_score_test = 0
    model.eval()

    with torch.no_grad():
        for all_data in test_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)
    #        y = y.float().unsqueeze(1).to(device=DEVICE)

            preds = torch.softmax(model(x), 1)
            preds = preds.cpu()
            preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            # preds = (preds = max_preds).float()
            preds = preds.cuda()

            num_correct_test += (preds == y).sum()
            num_pixels_test += torch.numel(preds)
            dice_score_test += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Test set acc:  {num_correct_test / num_pixels_test * 100:.2f} , dice score: {dice_score_test / len(test_loader)}"
    )

    with torch.no_grad():
        for all_data in train_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)
            #        y = y.float().unsqueeze(1).to(device=DEVICE)

            preds = torch.softmax(model(x), 1)
            preds = preds.cpu()
            preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            # preds = (preds = max_preds).float()
            preds = preds.cuda()

            num_correct_train += (preds == y).sum()
            num_pixels_train += torch.numel(preds)
            dice_score_train += (2 * (preds * y).sum()) / (
                      (preds + y).sum() + 1e-8 )
    print(
        f"Training set acc:  {num_correct_train/num_pixels_train*100:.2f} , dice score: {dice_score_train/len(train_loader)}"
    )
    model.train()
    return  num_correct_train/num_pixels_train, dice_score_train/len(train_loader), num_correct_test / num_pixels_test, dice_score_test / len(test_loader)

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, all_data in enumerate(loader):
        x = all_data[:, 0, :, :]
        y = all_data[:, 1:10, :, :]
        x = x.float().unsqueeze(1).to(device=DEVICE)
     #   y = y.float().unsqueeze(1)


        with torch.no_grad():
            preds = torch.softmax(model(x),1)
            preds = preds.cpu()
            preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            #preds = (preds = max_preds).float()
            preds = preds.cuda()

        for itr in range(9):
            torchvision.utils.save_image(
                preds[:, itr, :, :].unsqueeze(1), f"{folder}/pred_{idx}_itr_{itr}.png"
            )
            torchvision.utils.save_image(y[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_itr_{itr}.png")

    model.train()