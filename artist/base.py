from abc import ABC, abstractmethod


class Artist(ABC):
    @abstractmethod
    def plot_loss(self, train_loss_all, val_loss_all):
        pass

    @abstractmethod
    def plot_accuracy(self, train_acc_all, val_acc_all):
        pass

    @abstractmethod
    def plot_confusion_matrix(self, y, y_pred):
        pass
