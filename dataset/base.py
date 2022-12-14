import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Dataset(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def read_train(self, augment):
        pass

    @abstractmethod
    def read_test(self):
        pass

    @abstractmethod
    def read_captchas(self):
        pass

    @abstractmethod
    def create_dataLoader(self, X, y):
        pass
