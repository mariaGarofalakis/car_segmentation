import torch
from skimage import io, transform
import numpy as np


class Normalize(object):
    """Normalize image between 0-1
        """
    def __call__(self,sample):
        sample[0, :, :] = (sample[0, :, :] - np.min(sample[0, :, :])) / np.ptp(sample[0, :, :])
        sample[1, :, :] = (sample[1, :, :] - np.min(sample[1, :, :])) / np.ptp(sample[1, :, :])
        sample[2, :, :] = (sample[2, :, :] - np.min(sample[2, :, :])) / np.ptp(sample[2, :, :])

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        sample = np.transpose(sample, (1, 2, 0))
        image = sample[:, :, :3]
        masks = sample[:, :, 3:]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        masks = transform.resize(masks, (new_h, new_w))
        rescaled_sample = np.concatenate((image, masks), axis=2)

        return rescaled_sample

#class Grayscale(object):
#    """Convert RBG image in sample to Grayscale."""
#    def __call__(self, sample):
#        sample[:, :, :3] = np.dot(sample[:, :, :3], [0.299, 0.587, 0.144])
#        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = sample.transpose((2, 0, 1))
        return torch.from_numpy(sample)





