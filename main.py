import logging

import torch
from torch.utils.data import TensorDataset

from dataset.captcha import CaptchaDataset
from model.cnn import CNN
from artist.plots import Plots
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# setting device to GPU if available, CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# decide whether to load the saved model or retrain a new one
load_saved_model = False

# decide whether you want to plot Train accuracy&loss, and confusion matrix
plotting = True

# decide batch size for dataLoaders and model
batch_size = 256

# initialize Dataset object and Artist object
dataset = CaptchaDataset()
artist = Plots()

if not load_saved_model:

    logger.info("NOT loading saved model...")

    # decide whether you want to augment the training data when reading it
    # flip_horizontal, flip_vertical, rotate, gaussian noise, shear, gaussian blur
    augment_data = False

    # read training data
    logger.info("Reading training dataset...")
    X_train, y_train = dataset.read_train(augment=augment_data)
    logger.info("Done.")

    # create list of tuples from the training split
    logger.info("Creating training and validation splits...")
    train_split = map_features_and_labels(X_train, y_train)

    # create train and validation splits based on user selection
    if augment_data:
        train_split, validation_split = torch.utils.data.random_split(train_split, [450000, 110000],
                                                                      generator=torch.Generator().manual_seed(42))
    else:
        train_split, validation_split = torch.utils.data.random_split(train_split, [65000, 15000],
                                                                      generator=torch.Generator().manual_seed(42))

    logger.info("Done.")

    # check dataset details
    logger.info("Size after split...")
    logger.info(f"Train split: {len(train_split)}")
    logger.info(f"Validation split: {len(validation_split)}")

    # load the dataset into dataLoader
    logger.info("Creating train & validation DataLoaders...")
    train_iter = dataset.create_dataLoader(train_split, batch_size=batch_size, shuffle=True)
    validation_iter = dataset.create_dataLoader(validation_split, batch_size=batch_size)
    logger.info("Done.")

    # initialize Model object with desired hyper_parameters
    hyper_parameters = {'batch_size': batch_size, 'learning_rate': 0.001, 'epochs': 5}
    model = CNN(hyper_parameters=hyper_parameters)

    # create a network ~ this is a static method in the CNN class, for different architecture it should be modified
    # there (adding, removing layers etc)
    logger.info("Creating network...")
    net = model.create_network()
    logger.info("Done.")

    # train the model
    logger.info("Training network...")
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = model.train(net, train_iter, validation_iter, device=device)
    logger.info("Done.")

    if plotting:
        # if user wants to plot, plotting training loss & accuracy
        logger.info("Plotting training loss & accuracy...")
        artist.plot_loss(train_loss_all, val_loss_all)
        artist.plot_accuracy(train_acc_all, val_acc_all)
        logger.info("Done.")

else:

    # if user decides to load saved model
    model = CNN()
    logger.info("Loading saved model...")
    net = model.load('saved_models/model.pth')
    logger.info("Done.")


# reading test dataset and creating the list of tuples (value, label)
logger.info("Reading test dataset...")
X_test, y_test = dataset.read_test()
test_split = map_features_and_labels(X_test, y_test)
logger.info(f"Test split: {len(test_split)}")

# creating dataLoader for test dataset
test_iter = dataset.create_dataLoader(test_split, batch_size=batch_size)

# testing the network
logger.info("Testing network...")
test_loss, test_acc, y_pred = model.test(net, test_iter, plot=plotting, device=device)
logger.info(f'Test loss {test_loss:.2f}, Test accuracy {test_acc:.2f}')
logger.info("Done.")
logger.info("Saving model...")
model.save(net, 'saved_models/model.pth')
logger.info("Done.")

if plotting:
    # if user wants plots, plotting the test results confusion matrix
    logger.info("Plotting confusion matrix...")
    artist.plot_confusion_matrix(y_test, y_pred)
    logger.info("Done.")

# reading captchas dataset
logger.info("Reading captchas...")
X_captcha, y_captcha = dataset.read_captchas()
logger.info("Done.")

# predicting captchas using model (showing both full captcha accuracy and single digit accuracy
# <after splitting the big image into smalled one>)
logger.info("Predicting on captchas...")
captcha_acc, single_pred = model.predict(net, X_captcha, y_captcha, device=device)
logger.info(f'Captcha accuracy {captcha_acc:.2f}')
logger.info(f'Single digits accuracy {single_pred:.2f}')
logger.info("Done.")



