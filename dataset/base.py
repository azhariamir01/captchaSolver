import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Dataset(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def read_train(self):
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
