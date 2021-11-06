import torch
import torchvision
from dataset import create_dataset
from torch.utils.data import DataLoader



def get_loaders(train_dir, batch_size, train_transform, num_workers=4, pin_memory=True):

    train_ds = create_dataset(image_dir=train_dir, transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True)
    return train_loader