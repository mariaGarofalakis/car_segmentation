import os
from torch.utils.data import Dataset
import numpy as np
import torch

class create_dataset(Dataset):
    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        all_images = os.listdir(image_dir)
        self.original_images = [x for x in all_images if '_a.' in x]
        self.transform = transform


    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.original_images[index])
        image = np.load(image_path)



        if self.transform is not None:
            image = self.transform(image)

        return image