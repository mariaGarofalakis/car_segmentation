import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TRAIN_IMG_DIR = "C:/Users/maria/Desktop/mathimata/deep/project/car_segmentation_2021/clean_data"

def preprocessing():
    print('perform preprocessing')
    image_dir = TRAIN_IMG_DIR
    all_images = os.listdir(image_dir)
    original_images = [x for x in all_images if '_a.' in x]

    normalized_images = np.zeros((len(original_images),3,256, 256))

    for imgag_idx, original_image in enumerate(original_images):
        image_path = os.path.join(image_dir, original_image)
        image = np.load(image_path)

        normalized_images[imgag_idx, 0, :, :] = (image[0, :, :] - np.min(image[0, :, :])) / np.ptp(image[0, :, :])
        normalized_images[imgag_idx, 1, :, :] = (image[1, :, :] - np.min(image[1, :, :])) / np.ptp(image[1, :, :])
        normalized_images[imgag_idx, 2, :, :] = (image[2, :, :] - np.min(image[2, :, :])) / np.ptp(image[2, :, :])

    return normalized_images

def plot_images(images):

    img = images[1]
    img = np.transpose(img, (1, 2, 0))
    imgplot = plt.imshow(img)
    plt.show()





if __name__ == '__main__':
    normalized_images = preprocessing()
    plot_images(normalized_images)
