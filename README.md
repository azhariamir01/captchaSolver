
# Captcha detection

## Setup

Using URL:
```
git clone https://github.com/azhariamir01/captchaSolver
```
Using SSH:
```
git@github.com:azhariamir01/captchaSolver.git
```

# Project structure

## Dataset

- ```base.py``` -  base class Dataset(abstract)
- ```captcha.py``` - CaptchaDataset class with it's methods (read_train, read_test, read_captcha, augment_data, create_dataLoader)

## Model

- ```base.py``` - base class Model(abstract)
- ```cnn.py``` - CNN class with it's methods (create_network, train, train_epoch, evaluate_accuracy, test, predict, load, save)

## Artist

- ```base.py``` - base class Artist(abstract)
- ```plots.py``` - Plots class with it's methods (plot_accuracy, plot_loss, plot_confusion_matrix)

## Other

- ```utils.py``` - some functions used throughout the project
- ```main.py``` - main logic of the project and what to run

# Run

```pip install -r requirements.txt```

```python main.py```

### To keep in mind:

In ```main.py``` there are some parameters that change the overall logic of the project:

- ```load_saved_model``` ~ if True, it will load existing model, and skip the training part, if False it will proceed normally
- ```plotting``` ~ if True, it will plot the training&validation loss and accuracy, and the test confusion matrix, if False it will not
- ```augment_data``` ~ if True, it will augment the training data when reading it, going from 80k to 560k (see augmentation methods in code)
- ```batch_size``` ~ batch size used for the dataLoaders and for the model training.
      
