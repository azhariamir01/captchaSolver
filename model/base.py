from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, net, train_iter, validation_iter, device):
        pass

    @abstractmethod
    def test(self, net, data, plot, device):
        pass

    @abstractmethod
    def save(self, net, PATH):
        pass

    @abstractmethod
    def load(self, PATH):
        pass

    @abstractmethod
    def predict(self, net, X, y, device):
        pass
