def map_features_and_labels(X, y):
    """
    Creates a list of tuples from X and y
    :param X: list
    :param y: list
    :return: list of tuples (X[i], y[i])
    """
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])

    return data


def get_labels(labels):
    """
    Returns the initial label as string based on index 0-15 -> 0-9+A-F
    :param labels: list of labels
    :return: list of matching labels
    """
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    return [text_labels[i] for i in labels]
