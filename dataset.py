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

   #     image[0, :, :] = (image[0, :, :] + 0.485 / 0.229 - 0.485) / 0.229
   #     image[1, :, :] = (image[1, :, :] + 0.456 / 0.224 - 0.456) / 0.224
   #     image[2, :, :] = (image[2, :, :] + 0.406 / 0.225 - 0.406) / 0.225

        # Normalised [0,1]
        image[0, :, :] = (image[0, :, :] - np.min(image[0, :, :])) / np.ptp(image[0, :, :])
        image[1, :, :] = (image[1, :, :] - np.min(image[1, :, :])) / np.ptp(image[1, :, :])
        image[2, :, :] = (image[2, :, :] - np.min(image[2, :, :])) / np.ptp(image[2, :, :])

        final_image = torch.tensor(image[:3])



     #   if self.transform is not None:
      #      final_image = self.transform(final_image)

        return final_image