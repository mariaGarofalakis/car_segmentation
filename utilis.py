import torch
import torchvision
from dataset import create_dataset
from torch.utils.data import DataLoader
import TotalLoss
import csv
from os.path import exists
import matplotlib.pyplot as plt
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_metrics(name: str):
    data = []
    headers = None
    with open(name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data.append(row)
    train_metrics = []
    test_metrics = []
    for idx,row in enumerate(data):
        if idx == 0:
            headers = row
        elif (idx % 2) == 1:
            train_metrics.append(row)
        elif (idx % 2) == 0:
            test_metrics.append(row)

    epochs = []
    for i in range(len(train_metrics)):
        epochs.append(i)
    recall_train = []
    recall_test = []
    dice_train = []
    dice_test = []
    for i in train_metrics:
        recall_train.append( float("{:.2f}".format(float(i[0]))) )
    for i in test_metrics:
        recall_test.append(float("{:.2f}".format(float(i[0]))))
    for i in train_metrics:
        dice_train.append( float("{:.2f}".format(float(i[1]))) )
    for i in test_metrics:
        dice_test.append(float("{:.2f}".format(float(i[1]))))

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot( epochs , recall_train, label='train_loss')
    plt.plot( epochs, recall_test, label='valid_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot( epochs, dice_train, label='train_accs')
    plt.plot(epochs, dice_test, label='valid_accs')
    plt.legend()
    plt.savefig('../metrics/metrics_baseline.png')


def save_metrics(metrics,name: str):
    data = []
    file_exists = exists(name)
    header = ['Recal', 'Dice', 'Tversky', 'Cross Entropy', 'Total', 'IoU']
    for loader in metrics:
        for i in range(len(loader)):
            if i ==0:
                loader[i] =  float ( "{:.2f}".format( float( loader[i].cpu().numpy()) ) )
            else:
                loader[i] = float("{:.4f}".format(float(loader[i].cpu().numpy())))
        data.append(loader)
    with open(name, 'a',newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for i in data:
            writer.writerow(i)
    plot_metrics(name)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir , test_dir, batch_size, train_transform, test_transform, num_workers=4, pin_memory=True):
    train_ds = create_dataset(image_dir=train_dir, train=True, transform = train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)

    test_ds = create_dataset(
        image_dir=test_dir, train=False, transform = test_transform
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
    TP_test = 0.0
    FN_test = 0.0
    total_test = 0.0
    union_test = 0.0
    TP_test_modified = 0.0
    FN_test_modified = 0.0
    dice_score_test = 0.0
    cross_loss_test = 0.0
    total_loss_test = 0.0
    Tversky_test = 0.0

    TP_train = 0.0
    FN_train = 0.0
    total_train = 0.0
    TP_train_modified = 0.0
    FN_train_modified = 0.0
    dice_score_train = 0.0
    cross_loss_train = 0.0
    total_loss_train = 0.0
    Tversky_train = 0.0

    model.eval()

    #Set loss function
    total = TotalLoss.Total_loss()

    with torch.no_grad():
        for all_data in test_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)

            preds = model(x)
            loss = total(preds, y)
            total_loss_test += loss
            cross_loss_test += total.ce
            Tversky_test += total.tversky

            preds = torch.softmax(preds, 1)
            preds = (torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)).to(device=DEVICE)

            TP_test_modified += (preds[:, 1:9, :, :] * y[:, 1:9, :, :]).sum()
            FN_test_modified += (y[:, 1:9, :, :] * (1 - preds[:, 1:9, :, :])).sum()
            TP_test += (preds * y).sum()
            FN_test += (y * (1 - preds)).sum()
            total_test += (preds + y).sum()

            dice_score_test += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    union_test += total_test - TP_test
    print(
        f"Testing set:...Recall: {TP_test/(TP_test+FN_test)*100:.2f} ,...Car Recall: {TP_test_modified/(TP_test_modified+FN_test_modified)*100} ,...Dice score: {1-dice_score_test/len(test_loader)} "
        f",...Tversky loss: {Tversky_test/len(test_loader)},...Cross_entropy: {cross_loss_test/len(test_loader)}"
    )

    with torch.no_grad():
        for all_data in train_loader:
            x = all_data[:, 0, :, :]
            y = all_data[:, 1:10, :, :]
            x = x.float().unsqueeze(1).to(device=DEVICE)
            y = y.float().to(device=DEVICE)

            preds = model(x)
            loss = total(preds, y)
            total_loss_train += loss
            cross_loss_train += total.ce
            Tversky_train += total.tversky

            preds = torch.softmax(preds, 1)
            preds = (torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)).to(device=DEVICE)

            TP_train_modified += (preds[:, 1:9, :, :] * y[:, 1:9, :, :]).sum()
            FN_train_modified += (y[:, 1:9, :, :] * (1 - preds[:, 1:9, :, :])).sum()
            TP_train += (preds * y).sum()
            FN_train += (y * (1 - preds)).sum()
            total_train += (preds + y).sum()


            dice_score_train += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
        union_train = total_train - TP_train
        print(
            f"Training set:...Recall: {TP_train / (TP_train + FN_train) * 100:.2f} ,...Car Recall: {TP_train_modified/(TP_train_modified+FN_train_modified)*100} ,...Dice score: {1 - dice_score_train / len(train_loader)} "
            f",...Tversky loss: {Tversky_train / len(train_loader)},...Cross_entropy: {cross_loss_train / len(train_loader)}"
        )

    model.train()

    #Return [ [train_metris] , [test_metrics] ] = [ [Recall,Dice,Tversky(used),Cross-Entropy(used),Total(used),IoU, Recall_modified] , [Recall,...,Recall_modified] ]
    return [
                [(TP_train/(TP_train+FN_train))*100, 1 - (dice_score_train/len(train_loader)),
                 (Tversky_train/len(train_loader)),
                 (cross_loss_train/len(train_loader)), (total_loss_train/len(train_loader)),
                 (TP_train+1e-8)/(union_train+1e-8),
                 (TP_train_modified/(TP_train_modified+FN_train_modified))*100],
                [(TP_test / (TP_test + FN_test)) * 100, 1 - (dice_score_test / len(test_loader)),
                 (Tversky_test / len(test_loader)),
                 (cross_loss_test / len(test_loader)), (total_loss_test / len(test_loader)),
                 (TP_test + 1e-8) / (union_test + 1e-8),
                 (TP_test_modified/(TP_test_modified+FN_test_modified))*100]
            ]


def check_top_five(dir, name, metric, model_snapshot):
    dictionary = {
            "checkpoint_0": float('inf'),
            "checkpoint_1": float('inf'),
            "checkpoint_2": float('inf'),
            "checkpoint_3": float('inf'),
            "checkpoint_4": float('inf')
    }
    if not exists(dir+name):
        json_object = json.dumps(dictionary)
        with open(dir+name, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    f = open(dir+name, 'r')
    data = json.load(f)
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    checkpoint = next(iter(data))
    print(f"{data[checkpoint]} > {metric}????")
    if data[checkpoint] > metric:
        data[checkpoint] = metric
        save_checkpoint(model_snapshot, filename=dir+checkpoint+".pth.tar")
    else:
        print("Checkpoint is not in top 5 :'(")
    f.close()

    with open(dir + name, "w") as file:
        json_object = json.dumps(data)
        file.write(json_object)
    file.close()


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
            #preds = torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1), 1.0)
            #preds = (preds = max_preds).float()
            preds = preds.cuda()

        for itr in range(9):
            torchvision.utils.save_image(
                torch.zeros(preds.shape).scatter(1, preds.argmax(1).unsqueeze(1).cpu(), 1.0)[:, itr, :, :].unsqueeze(1).cuda(), f"{folder}/pred_{idx}_itr_{itr}.png"
            )
            torchvision.utils.save_image(preds[:, itr, :, :].unsqueeze(1), f"{folder}{idx}_grey_{itr}.png")

    model.train()