import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

from sklearn.metrics import mean_squared_error


def load_model(filepath='saved_model/model.h5'):
    """
    Loads a model from a filepath
    """
    model = tf.keras.models.load_model(filepath)
    return model


def predicting_model(model, X, y=None, ndim='default'):
    """
    Predicts the output of a model
    ndim: {'default', 1} default is: 'default'
    """
    if y is None:
        if ndim == 'default':
            return model.predict(X)
        return np.squeeze(model.predict(X))
    if ndim == 'default':
        return model.predict([[X, y]])
    return np.squeeze(model.predict([[X, y]]))


def score(real, predicted):
    """
    Calculates the accuracy of a model
    """

    if not isinstance(real, np.ndarray):
        real = np.array([real])
    if not isinstance(predicted, np.ndarray):
        predicted = np.array([predicted])
    return mean_squared_error(real, predicted)


def print_accuracy(real, predicted):
    """
    Prints the accuracy of a model
    """
    return f"Mean Squared Error: {score(real, predicted):.4f}"
