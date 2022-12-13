from typing import Dict, Optional

import imutils
import numpy as np

from model.base import Model
import torch
from torch import nn
from utils import *
import logging
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CNN(Model):
    def __init__(self, hyper_parameters: Optional[Dict] = None):
        super().__init__(name="cnn")

        if hyper_parameters:
            self.batch_size = hyper_parameters['batch_size']
            self.learning_rate = hyper_parameters['learning_rate']
            self.epochs = hyper_parameters['epochs']
        else:
            self.batch_size = 32
            self.learning_rate = 0.01
            self.epochs = 10

        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def create_network():
        net = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(50, 80, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 500),
            nn.ReLU(),
            nn.Linear(500, 16)
        )

        return net

    @staticmethod
    def train_epoch(net, train_iter, loss, optimizer, device):
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
    def evaluate_accuracy(net, data_iter, device, loss=nn.CrossEntropyLoss()):
        """Compute the accuracy for a model on a dataset."""
        net.eval()  # Set the model to evaluation mode

        total_loss = 0
        total_hits = 0
        total_samples = 0
        with torch.no_grad():
            for X, y in data_iter:
                # type cast the input
                X = X.type(torch.float32)
                y = y.type(torch.LongTensor)

                # send input to device
                X = X.to(device)
                y = y.to(device)

                y_hat = net(X)
                l = loss(y_hat, y)
                total_loss += float(l)
                total_hits += sum(net(X).argmax(axis=1).type(y.dtype) == y)
                total_samples += y.numel()
        return float(total_loss) / len(data_iter), float(total_hits) / total_samples * 100

    def train(self, net, train_iter, validation_iter, device='cpu'):
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
            val_loss, val_acc = self.evaluate_accuracy(net, validation_iter, device, self.loss)
            val_loss_all.append(val_loss)
            val_acc_all.append(val_acc)

            logger.info(f"Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation "
                        f"loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}")

        return train_loss_all, train_acc_all, val_loss_all, val_acc_all

    def test(self, net, data, device='cpu'):
        """Predict labels."""

        test_loss, test_acc = self.evaluate_accuracy(net, data, device)
        return test_loss, test_acc

    def save(self, net, PATH):
        torch.save(net, PATH)

    def load(self, PATH):
        return torch.load(PATH)

    def predict(self, net, X, y, device='cpu'):
        true_pred = 0
        map_labels = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15'}

        net.eval()
        net.to(device)

        for counter, image in enumerate(X):

            split_images = [image[:24, :], image[24:48, :], image[48:72, :], image[72:, :]]

            predictions = []
            label = y[counter]

            for split_image in split_images:

                split_image = np.expand_dims(split_image, axis=0)
                split_image = np.expand_dims(split_image, axis=0)
                split_image = torch.from_numpy(split_image)
                split_image = split_image.type(torch.float32)

                split_image = split_image.to(device)

                output = net(split_image)
                y_pred = torch.argmax(output).item()

                if str(y_pred) in map_labels.values():
                    key = [k for k, v in map_labels.items() if v == str(y_pred)]
                    predictions.append(key[0])
                else:
                    predictions.append(str(y_pred))

            captcha_pred = "".join(predictions)
            if captcha_pred == label:
                true_pred += 1

        return float(true_pred) / len(X)
