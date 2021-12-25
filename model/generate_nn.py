import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


class Train:

    def __init__(self,):
        pass

    def define_model(self, activation_func=None, final_activation=None,):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(
            units=8, input_shape=[2], activation=activation_func))
        model.add(tf.keras.layers.Dense(units=1, activation=final_activation))
        return model

    def train_model(self, X, y, model=None, epochs=10, validation_split=0,
                    verbose=0, loss='mae', optimizer='SGD', metrics=[None],):
        if model is None:
            model = self.define_model()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(X, y, epochs=epochs,
                  validation_split=validation_split, verbose=verbose)
        return model

    def predict_model(self, model, X):
        return model.predict(X)
