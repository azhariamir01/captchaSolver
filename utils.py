import imutils
import cv2
import torch.utils as utils


def map_features_and_labels(X, y):
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])

    return data


def get_labels(labels):
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    return [text_labels[i] for i in labels]
