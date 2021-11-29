import torch
import torch.nn as nn
import torchvision
from dataset import create_dataset
from torch.utils.data import DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def save_checkpoint_background(state, filename="remove_background_pretrained_model.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def remove_background(data,model2):

    model2.eval()
    the_car = data[:, 0, :, :].float().unsqueeze(1)

    predictions_remove_background = model2(the_car)
    predictions_remove_background = torch.sigmoid(predictions_remove_background)
    predictions_remove_background = (predictions_remove_background > 0.5).float()
    for itr in range(11):
        if itr != 1:
            new_data = (data[:, itr, :, :] > 0.5).float().unsqueeze(1)
            filtered = torch.squeeze(new_data * predictions_remove_background)
            data[:, itr, :, :] = filtered

    return data

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
    TP_test=0
    FN_test=0
    TP_train = 0
    FN_train = 0
    dice_score_train = 0
    dice_score_test = 0
    model.eval()

    with torch.no_grad():
        for all_data in test_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)

            preds = torch.softmax(model(x), 1)
            preds = (torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)).to(device=DEVICE)

            TP_test += (preds[:, 1:9, :, :] * y[:, 1:9, :, :]).sum()
            FN_test += (y[:, 1:9, :, :] * (1 - preds[:, 1:9, :, :])).sum()
            dice_score_test += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

            class_weight = torch.tensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1])
            closs = nn.CrossEntropyLoss(weight= class_weight.to(device=DEVICE), reduction='mean')
            cross_loss = closs(preds,y)
            # flatten label and prediction tensors
            inputs_f = preds.view(-1)
            targets_f = y.view(-1)
            inputs = preds[:, 1:9, :, :].reshape(-1)
            targets = y[:, 1:9, :, :].reshape(-1)
            TP = (inputs * targets).sum()
            FP = ((1 - targets_f) * inputs_f).sum()
            FN = (targets_f * (1 - inputs_f)).sum()
            Tversky = 1 - ( TP + 1e-4) / (TP + 0.3*FP + 0.7*FN + 1e-4)

    print(
        f"Testing set:....Recall: {TP_test/(TP_test+FN_test)*100:.2f} ,...Dice score: {1-dice_score_test/len(test_loader)} ,...Tversky loss: {0.6*Tversky},...Cross_entropy: {0.4*cross_loss}"
    )

    with torch.no_grad():
        for all_data in train_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)

            preds = torch.softmax(model(x), 1)
            preds = (torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)).to(device=DEVICE)

            TP_train += (preds[:,1:9,:,:] * y[:,1:9,:,:]).sum()
            FN_train += (y[:,1:9,:,:] * (1 - preds[:,1:9,:,:])).sum()
            dice_score_train += (2 * (preds * y).sum()) / (
                      (preds + y).sum() + 1e-8 )

            cross_loss = closs(preds, y)
            # flatten label and prediction tensors
            inputs_f = preds.view(-1)
            targets_f = y.view(-1)
            inputs = preds[:, 1:9, :, :].reshape(-1)
            targets = y[:, 1:9, :, :].reshape(-1)
            TP = (inputs * targets).sum()
            FP = ((1 - targets_f) * inputs_f).sum()
            FN = (targets_f * (1 - inputs_f)).sum()
            Tversky = 1 - ( TP + 1e-4) / (TP + 0.3*FP + 0.7*FN + 1e-4)


    print(
        f"Training set:...Recall: {TP_train/(TP_train+FN_train)*100:.2f} , dice score: {1-dice_score_train/len(train_loader)},...Tversky loss: {0.6*Tversky},...Cross_entropy: {0.4*cross_loss}"
    )
    model.train()
    return   [ (TP_train/(TP_train+FN_train)).cpu(), (dice_score_train/len(train_loader)).cpu(), (TP_test/(TP_test+FN_test)).cpu(), (dice_score_test / len(test_loader)).cpu() ]


def check_accuracy_background(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for all_data in loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().unsqueeze(1).to(device=DEVICE)


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

def save_imgs_of_car_removing_background(loader, model2, folder="saved_no_back_images/", device=DEVICE):
    model2.eval()
    for idx, all_data in enumerate(loader):
        new_data = remove_background(all_data, model2)
        x = all_data[:, 0, :, :]
        y = all_data[:, 1:10, :, :]
        x = x.float().unsqueeze(1).to(device=DEVICE)
        #   y = y.float().unsqueeze(1)

        with torch.no_grad():
            preds = torch.softmax(model2(x), 1)
            preds = preds.cpu()
            preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            # preds = (preds = max_preds).float()
            preds = preds.cuda()

        for itr in range(9):
            torchvision.utils.save_image(
                preds[:, itr, :, :].unsqueeze(1), f"{folder}/pred_{idx}_itr_{itr}.png"
            )
            torchvision.utils.save_image(y[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_itr_{itr}.png")

    model2.train()


def save_predictions_as_imgs(
    loader, model,model2, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, all_data in enumerate(loader):
        new_data = remove_background(all_data, model2)
        x = all_data[:, 0, :, :]
        y = all_data[:, 1:10, :, :]
        x = x.float().unsqueeze(1).to(device=DEVICE)
     #   y = y.float().unsqueeze(1)


        with torch.no_grad():
            preds = torch.softmax(model(x),1)
            preds = preds.cpu()
            #preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            #preds = (preds = max_preds).float()
            preds = preds.cuda()

        for itr in range(9):
            torchvision.utils.save_image(
                torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)[:, itr, :, :].unsqueeze(1).cuda(), f"{folder}/pred_{idx}_itr_{itr}.png"
            )
            torchvision.utils.save_image(preds[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_grey_{itr}.png")

    model.train()


