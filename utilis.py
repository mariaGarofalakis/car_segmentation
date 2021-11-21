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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for all_data in loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 2:, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)
    #        y = y.float().unsqueeze(1).to(device=DEVICE)


            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, all_data in enumerate(loader):
        x = all_data[:, 0, :, :]
        y = all_data[:, 2:, :, :]
        x = x.float().unsqueeze(1).to(device=DEVICE)
     #   y = y.float().unsqueeze(1)


        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        for itr in range(9):
            torchvision.utils.save_image(
                preds[:, itr, :, :].unsqueeze(1), f"{folder}/pred_{idx}_itr_{itr}.png"
            )
            torchvision.utils.save_image(y[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_itr_{itr}.png")

    model.train()