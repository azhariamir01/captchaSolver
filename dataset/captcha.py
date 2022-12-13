import glob
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import os.path

from dataset.base import Dataset


class CaptchaDataset(Dataset):
    def __init__(self):
        super().__init__(name="captcha_dataset")

        self.training_path = "data/training_set"
        self.testing_path = "data/test_sets"
        self.captcha_path = "data/full_captchas"

    def read_train(self):
        possible_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}
        # possible_labels = ['0']
        data = []
        labels = []

        for label in possible_labels:
            new_path = self.training_path + '/' + label + '/*.png'
            for image_file in glob.glob(new_path):

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

    def read_test(self):
        possible_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        possible_test_sets = ['TS_F', 'TS_S', 'TS_V']
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}
        data = []
        labels = []

        for test_set in possible_test_sets:
            for label in possible_labels:
                new_path = self.testing_path + '/' + test_set + '/' + label + '/*.png'
                for image_file in glob.glob(new_path):

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

        data = []
        labels = []
        path = self.captcha_path + '/*.png'

        for image_file in glob.glob(path):

            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (32, 96))

            data.append(image)
            labels.append(image_file[23:27])

        return data, labels

    def create_dataLoader(self, data, batch_size=1, shuffle=False):
        dataLoader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return dataLoader
