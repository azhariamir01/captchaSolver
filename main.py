import logging

import torch
from torch.utils.data import TensorDataset

from dataset.captcha import CaptchaDataset
from model.cnn import CNN
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_saved_model = True
dataset = CaptchaDataset()

if not load_saved_model:

    logger.info("NOT loading saved model...")

    logger.info("Reading dataset...")
    X_train, y_train = dataset.read_train()
    X_test, y_test = dataset.read_test()
    logger.info("Done.")

    logger.info("Creating splits...")
    train_split = map_features_and_labels(X_train, y_train)
    train_split, validation_split = torch.utils.data.random_split(train_split, [70000, 10000],
                                                                  generator=torch.Generator().manual_seed(42))
    test_split = map_features_and_labels(X_test, y_test)
    logger.info("Done.")

    logger.info("Size after split...")
    logger.info(f"Train split: {len(train_split)}")
    logger.info(f"Validation split: {len(validation_split)}")
    logger.info(f"Test split: {len(test_split)}")

    logger.info("Creating DataLoader objects...")
    train_iter = dataset.create_dataLoader(train_split, batch_size=32, shuffle=True)
    validation_iter = dataset.create_dataLoader(validation_split, batch_size=32)
    test_iter = dataset.create_dataLoader(test_split, batch_size=32)
    logger.info("Done.")

    hyper_parameters = {'batch_size': 32, 'learning_rate': 0.001, 'epochs': 10}
    model = CNN(hyper_parameters=hyper_parameters)

    logger.info("Creating network...")
    net = model.create_network()
    logger.info("Done.")

    logger.info("Training network...")
    model.train(net, train_iter, validation_iter, device='cuda')
    logger.info("Done.")

    logger.info("Testing network...")
    test_loss, test_acc = model.test(net, test_iter, device='cuda')
    logger.info(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')
    logger.info("Done.")
    logger.info("Saving model...")
    model.save(net, 'saved_models/model.pth')
    logger.info("Done.")

else:
    model = CNN()
    logger.info("Loading saved model...")
    net = model.load('saved_models/model.pth')
    logger.info("Done.")


logger.info("Reading captchas...")
X_captcha, y_captcha = dataset.read_captchas()
logger.info("Done.")

logger.info("Predicting on captchas...")
captcha_acc = model.predict(net, X_captcha, y_captcha, device='cuda')
logger.info(f'Captcha accuracy {captcha_acc:.2f}')





