import glob
from torch.utils.data import DataLoader
import cv2
import numpy as np
from imgaug import augmenters as iaa

from dataset.base import Dataset


class CaptchaDataset(Dataset):
    def __init__(self):
        super().__init__(name="captcha_dataset")

        self.training_path = "data/training_set"
        self.testing_path = "data/test_sets"
        self.captcha_path = "data/full_captchas"

    def read_train(self, augment=False):
        """
        Reads training data
        :param augment: bool, default = False, user decision whether he wants to augment the train dataset or not
        :return: 2 np arrays, images and labels
        """
        possible_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}
        # possible_labels = ['0']
        data = []
        labels = []

        for label in possible_labels:
            new_path = self.training_path + '/' + label + '/*.png'
            for image_file in glob.glob(new_path):

                # we constructed the path from where we read and are going through all the png files from there

                # read image, convert to grayscale and resize to desired size
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (32, 24))

                if augment:

                    # augment the image, returns a list of 7 images, the original and 6 augmented
                    aug_images = self.data_augmentation(image)

                    # save the label so we can save it 7 times(amount of augmented images we have)
                    if label in map_labels.keys():
                        aug_label = int(map_labels[label])
                    else:
                        aug_label = int(label)

                    # save the labels and add another dimension to all images(original plus the augmented ones)
                    for i in range(0, len(aug_images)):
                        aug_images[i] = np.expand_dims(aug_images[i], axis=0)
                        labels.append(aug_label)

                    # add all images(including augmented ones to data)
                    data.extend(aug_images)

                else:
                    # expand dimension to make pytorch happy & append it to data, and matching label to the labels list
                    image = np.expand_dims(image, axis=0)
                    data.append(image)

                    if label in map_labels.keys():
                        labels.append(int(map_labels[label]))
                    else:
                        labels.append(int(label))

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        return data, labels

    def read_test(self):
        """
        Reads test data. This function combines TS_F, TS_V and TS_S.
        :return: 2 np arrays, images and labels
        """
        possible_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        # possible_test_sets = ['TS_F', 'TS_S', 'TS_V']
        possible_test_sets = ['TS_V']           # if desired to get only x of the 3 folders
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}
        data = []
        labels = []

        for test_set in possible_test_sets:
            for label in possible_labels:
                new_path = self.testing_path + '/' + test_set + '/' + label + '/*.png'
                for image_file in glob.glob(new_path):

                    # similarly, we keep constructing the path from where we want to read the test data
                    image = cv2.imread(image_file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (32, 24))

                    image = np.expand_dims(image, axis=0)
                    data.append(image)

                    if label in map_labels.keys():
                        labels.append(int(map_labels[label]))
                    else:
                        labels.append(int(label))

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)
        return data, labels

    def read_captchas(self):
        """
        Reads captcha images
        :return: 2 lists, list of images and list of corresponding labels
        """
        data = []
        labels = []
        path = self.captcha_path + '/*.png'

        for image_file in glob.glob(path):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (32, 96))

            data.append(image)
            # we know the format of the image name, thus we know the length of it, so we can easily extract labels
            labels.append(image_file[23:27])

        return data, labels

    @staticmethod
    def data_augmentation(image):
        """
        This function takes as input an image, augments it, and returns a list with the image + augmented images
        Augments:
            -horizontal flip
            -vertical flip
            -random rotation
            -Gaussian noise
            -shear
            -Gaussian blur
        :param image: Image type file
        :return: list of initial image as first element, followed by augmented images
        """
        images_aug = [image]

        # Flips, rotate
        flip_horizontal = iaa.Fliplr(1)  # % of images
        flip_vertical = iaa.Flipud(1)  # % of images
        rotate = iaa.Affine(rotate=(-45, 45))  # Random between (-25, 25)

        # Noise
        gnoise = iaa.AdditiveGaussianNoise(scale=(0, .4 * 255))
        # salt_pepper = iaa.SaltAndPepper(0.2)  # % of pixels

        # Cut
        # cutout = iaa.Cutout(nb_iterations=2)

        # Shearing
        shear = iaa.Affine(shear=(-16, 16))

        # Blur
        gblur = iaa.GaussianBlur(sigma=(0., 6.))
        # avgblur = iaa.AverageBlur(k=(2, 11))

        effects = [flip_horizontal, flip_vertical, rotate, gnoise, shear, gblur]

        for effect in effects:
            img_aug = effect(image=image)
            images_aug.append(img_aug)

        return images_aug

    def create_dataLoader(self, data, batch_size=1, shuffle=False):
        """
        Created data loader from dataset
        :param data: dataset
        :param batch_size: int, default = 1
        :param shuffle: bool, default = False
        :return: dataLoader object
        """
        dataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return dataLoader
