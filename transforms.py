# Imports
import numpy as np
import cv2
import torch
from skimage import transform

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


class randomHueSaturationValue(object):

    def __init__(self, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):

        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.u = u

    def __call__(self,samples):

        image = samples[:,:,:3]

        image = np.array(image)
        image = image[:, :, ::-1].copy()

        if np.random.random() < self.u:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        image = image[:, :, ::-1].copy()

        samples[:,:,:3] = image

        return samples

class randomShiftScaleRotate(object):

    def __init__(self,shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.borderMode = borderMode
        self.u = u


    def __call__(self,samples):


        image = np.array(samples[:,:,:3])
        mask = np.array(samples[:,:,3:])
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        if np.random.random() < self.u:
            height, width, channel = image.shape

            angle = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1])  # degree
            scale = np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1])
            aspect = np.random.uniform(1 + self.aspect_limit[0], 1 + self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.borderMode,
                                        borderValue=(0, 0, 0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=self.borderMode,
                                       borderValue=(0, 0, 0,))
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)

        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        samples[:, :, :3]=image
        samples[:, :, 3:] = mask

        return samples

class randomHorizontalFlip(object):

    def __init__(self, u=0.5):
        self.u = u

    def __call__(self,samples):
        image = np.array(samples[:,:,:3])
        mask = np.array(samples[:,:,3:])
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        if np.random.random() < self.u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        samples[:, :, :3] = image
        samples[:, :, 3:] = mask

        return samples


class randomZoom(object):
    def __init__(self, zoom_limit=0.25, u=0.5):

        self.zoom_limit = zoom_limit
        self.u = u

    def __call__(self,samples):
        image = np.array(samples[:,:,:3])
        mask = np.array(samples[:,:,3:])
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        if np.random.random() < self.u:
            value = np.random.uniform(self.zoom_limit, 1)
            h, w = image.shape[:2]
            h_taken = int(value * h)
            w_taken = int(value * w)
            h_start = np.random.randint(0, h - h_taken)
            w_start = np.random.randint(0, w - w_taken)
            image = image[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
            mask = mask[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
            image = cv2.resize(image, (h, w), cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (h, w), cv2.INTER_CUBIC)

        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

        samples[:, :, :3] = image
        samples[:, :, 3:] = mask

        return samples