# Imports
import numpy as np
import cv2

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):

    image = np.array(image)
    image = image[:, :, ::-1].copy()

    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    image = image[:, :, ::-1].copy()

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    return image, mask

def randomZoom(image, mask, zoom_limit=0.25, u=0.5):

    image = np.array(image)
    mask = np.array(mask)
    image = image[:, :, ::-1].copy()
    mask = mask[:, :, ::-1].copy()

    if np.random.random() < u:
        value = np.random.uniform(zoom_limit, 1)
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

    return image, mask