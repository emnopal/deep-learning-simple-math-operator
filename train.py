import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from dataset.generator import generate_dataset
from model.generate_nn import Train


# Define custom models
class CustomModel(Train):
    def define_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            256, input_shape=[2], activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model


# Addition
def addition_trained_model():
    X, y = generate_dataset(operation='add')
    train = Train()
    model = train.define_model()
    model = train.train_model(X, y, model, epochs=100,
                              verbose=1, optimizer='adam', validation_split=.5)
    if not os.path.isdir('saved_model'):
        os.mkdir('saved_model')
    if not os.path.isfile('saved_model/addition.h5'):
        model.save('saved_model/addition.h5')
    return


# Subtraction
def subtraction_trained_model():
    X, y = generate_dataset(operation='subtract')
    train = Train()
    model = train.define_model()
    model = train.train_model(X, y, model, epochs=100,
                              verbose=1, optimizer='adam', validation_split=.5)
    if not os.path.isdir('saved_model'):
        os.mkdir('saved_model')
    if not os.path.isfile('saved_model/subtract.h5'):
        model.save('saved_model/subtract.h5')
    return


# Multiplication
def multiplication_trained_model():
    X, y = generate_dataset(operation='multiply', include_zeros=False)
    multiply_train = CustomModel()
    model = multiply_train.define_model()
    model = multiply_train.train_model(X, y, model, epochs=500, verbose=1,
                                       validation_split=.3,
                                       optimizer=tf.optimizers.Adam(
                                           learning_rate=1e-4)
                                       )
    if not os.path.isdir('saved_model'):
        os.mkdir('saved_model')
    if not os.path.isfile('saved_model/multiplication.h5'):
        model.save('saved_model/multiplication.h5')
    return


# Divide
def divide_trained_model():
    X, y = generate_dataset(operation='divide', include_zeros=False)
    divide_train = CustomModel()
    model = divide_train.define_model()
    model = divide_train.train_model(X, y, model, epochs=500, verbose=1,
                                       validation_split=.3,
                                       optimizer=tf.optimizers.Adam(
                                           learning_rate=1e-4)
                                       )
    if not os.path.isdir('saved_model'):
        os.mkdir('saved_model')
    if not os.path.isfile('saved_model/divide.h5'):
        model.save('saved_model/divide.h5')
    return


if __name__ == '__main__':
    addition_trained_model()
    subtraction_trained_model()
    multiplication_trained_model()
    divide_trained_model()
