from artist.base import Artist
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Plots(Artist):
    def plot_accuracy(self, train_acc_all, val_acc_all):
        """
        This function plots the training and validation accuracy
        :param train_acc_all: training accuracy
        :param val_acc_all: validation accuracy
        :return:
        """

        epochs = range(1, len(train_acc_all) + 1)
        plt.plot(epochs, train_acc_all, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_all, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_loss(self, train_loss_all, val_loss_all):
        """
        This function plots training and validation loss
        :param train_loss_all: training loss
        :param val_loss_all: validation loss
        :return:
        """
        epochs = range(1, len(train_loss_all) + 1)
        plt.plot(epochs, train_loss_all, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_all, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y, y_pred):
        """
        This function plots the confusion matrix based on test predictions
        :param y: Ground truth test labels
        :param y_pred: Predicted test labels
        :return:
        """

        ConfusionMatrixDisplay.from_predictions(y, y_pred)
        plt.show()
