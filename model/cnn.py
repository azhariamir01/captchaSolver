import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from model.base import Model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CNN(Model):
    def __init__(self, hyper_parameters: Optional[Dict] = None):
        super().__init__(name="cnn")

        # if we receive the dictionary as a parameter extract the hyper_params from it, else use default ones
        if hyper_parameters:
            self.batch_size = hyper_parameters['batch_size']
            self.learning_rate = hyper_parameters['learning_rate']
            self.epochs = hyper_parameters['epochs']
        else:
            self.batch_size = 128
            self.learning_rate = 0.001
            self.epochs = 10

        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def create_network():
        """
        This function creates a neural network with the desired architecture
        :return: nn.net type
        """

        net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 1)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7680, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

        return net

    @staticmethod
    def train_epoch(net, train_iter, loss, optimizer, device):
        """
        This function trains an epoch
        :param net: network to be trained
        :param train_iter: batch_size set of training data (X, y)
        :param loss: desired loss function for the training
        :param optimizer: desired optimizer for the training
        :param device: desired device to perform the training ('cuda' or 'cpu')
        :return: training loss & training accuracy (float)
        """

        # Set the model to training mode
        net.train()
        # Sum of training loss, sum of training correct predictions, no. of examples
        total_loss = 0
        total_hits = 0
        total_samples = 0

        for X, y in train_iter:
            # type cast the input
            X = X.type(torch.float32)
            y = y.type(torch.LongTensor)

            # send input to device
            X = X.to(device)
            y = y.to(device)

            # Compute gradients and update parameters
            y_hat = net(X)
            l = loss(y_hat, y)
            # Using PyTorch built-in optimizer & loss criterion
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            total_loss += float(l)
            total_hits += sum(y_hat.argmax(axis=1).type(y.dtype) == y)
            total_samples += y.numel()
        # Return training loss and training accuracy
        return float(total_loss) / len(train_iter), float(total_hits) / total_samples * 100

    @staticmethod
    def evaluate_accuracy(net, data_iter, device, loss=nn.CrossEntropyLoss(), plot=False):
        """
        This function evaluates model accuracy on a dataset
        :param net: network desired to be evaluated
        :param data_iter: dataset to evaluate
        :param device: desired device to perform the training ('cuda' or 'cpu')
        :param loss: desired loss function for the training
        :param plot: bool whether or not the user wants to print the confusion matrix (in which case it saves individual
         predictions)
        :return: training loss, training accuracy (both float), and y_pred, list of individual predictions
        """
        net.eval()  # Set the model to evaluation mode

        total_loss = 0
        total_hits = 0
        total_samples = 0
        y_pred = []
        with torch.no_grad():
            for X, y in data_iter:
                # type cast the input
                X = X.type(torch.float32)
                y = y.type(torch.LongTensor)

                # send input to device
                X = X.to(device)
                y = y.to(device)

                # compute gradients and update parameters
                y_hat = net(X)
                l = loss(y_hat, y)
                total_loss += float(l)
                total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)

                # save individual predictions if user wants to plot the confusion matrix
                if plot:
                    for i in range(0, len(y_hat)):
                        y_pred.append(y_hat[i].argmax().item())

                total_samples += y.numel()
        return float(total_loss) / len(data_iter), float(total_hits) / total_samples * 100, y_pred

    def train(self, net, train_iter, validation_iter, device='cpu'):
        """
        This function envelops the training process that happens in train_epoch, and the evaluation process that
        happens in evaluate_accuracy
        :param net: network to be trained
        :param train_iter: train dataset
        :param validation_iter: validation dataset
        :param device: desired device
        :return: overall train loss&accuracy and validation loss&accuracy
        """
        train_loss_all = []
        train_acc_all = []
        val_loss_all = []
        val_acc_all = []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        net.to(device)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(net, train_iter, self.loss, optimizer, device)
            train_loss_all.append(train_loss)
            train_acc_all.append(train_acc)
            val_loss, val_acc, _ = self.evaluate_accuracy(net, validation_iter, device, self.loss)
            val_loss_all.append(val_loss)
            val_acc_all.append(val_acc)

            logger.info(f"Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation "
                        f"loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}")

        return train_loss_all, train_acc_all, val_loss_all, val_acc_all

    def test(self, net, data, plot, device='cpu'):
        """
        Tests the network, envelops function evaluate_accuracy
        :param net: network
        :param data: dataset
        :param plot: whether or not the user wants to plot the confusion matrix after testing
        :param device: desired device
        :return:
        """

        test_loss, test_acc, y_pred = self.evaluate_accuracy(net, data, device, plot=plot)
        return test_loss, test_acc, y_pred

    def save(self, net, PATH):
        """
        Saves the current state of the model to PATH
        :param net: model to be saved
        :param PATH: desired location to be saved
        :return:
        """
        torch.save(net, PATH)

    def load(self, PATH):
        """
        Loads last saved state of the model from PATH
        :param PATH: desired location from when to load the model
        :return: loaded network
        """
        return torch.load(PATH)

    def predict(self, net, X, y, device='cpu'):
        """
        Predicts full_captchas using the previously trained model
        :param net: network
        :param X: list of captcha images
        :param y: list of captcha labels
        :param device: desired device
        :return: overall captcha accuracy, and single digit captcha accuracy (both float)
        """
        true_pred = 0
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}

        net.eval()
        net.to(device)

        single_pred = 0

        for counter, image in enumerate(X):

            # split the image equally into 4 parts
            # I know this is the naive way to think of it, but unfortunately I am no openCV guru
            split_images = [image[:24, :], image[24:48, :], image[48:72, :], image[72:, :]]

            predictions = []
            label = y[counter]

            for split_image in split_images:

                # for each of the split images, we expand dims(to fit expected model input) & convert it to float tensor
                split_image = np.expand_dims(split_image, axis=0)
                split_image = np.expand_dims(split_image, axis=0)
                split_image = torch.from_numpy(split_image)
                split_image = split_image.type(torch.float32)

                split_image = split_image.to(device)

                # get prediction for split image
                output = net(split_image)
                y_pred = torch.argmax(output).item()

                # based on prediction get label and add it the predictions list
                if str(y_pred) in map_labels.values():
                    key = [k for k, v in map_labels.items() if v == str(y_pred)]
                    predictions.append(key[0])
                else:
                    predictions.append(str(y_pred))

            # combine the predictions list into a string and check if it matches ground truth label
            captcha_pred = "".join(predictions)
            if captcha_pred == label:
                true_pred += 1

            # for each prediction we check how many individual letters it got right, so we also know single digit acc
            for i in range(0, 4):
                if captcha_pred[i] == label[i]:
                    single_pred += 1

        return float(true_pred) / len(X), float(single_pred) / (4*len(X))
