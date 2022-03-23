from typing import Dict, List, Optional, Tuple

from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, SimpleRNN

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class AirQualityUciRNN():
    """Air quality uci RNN class"""

    def __init__(self) -> None:
        self.__model = None
        self.__air_quality_uci_data = None

    @property
    def air_quality_uci_data(self) -> object:
        """air_quality_uci_data getter

        :return object: AirQualityUciDataset object or None
        """
        return self.__air_quality_uci_data

    @property
    def model(self) -> Optional[Sequential]:
        """model getter

        :return Optional[Sequential]: air quality uci RNN getter or None
        """
        return self.__model

    def prepare_to_train(self, air_quality_uci_data: object, train_size: float, time_steps: int) -> None:
        """Run all process to prepare to train.

        :param object air_quality_uci_data: AirQualityUciDataset obejct to use training
        :param float train_size: ratio of training dataset size
        :param int time_steps: the length of data to predict
        """
        self.__set_dataset(air_quality_uci_data=air_quality_uci_data)
        self.__scale_data()
        self.__split_dataset(train_size=train_size)
        self.__split_all_dataset_into_x_and_y(time_steps=time_steps)

    def __set_dataset(self, air_quality_uci_data) -> None:
        """Set air_quality_uci_data.
        """
        self.__air_quality_uci_data = air_quality_uci_data

    def __scale_data(self) -> None:
        """Scale data to train.
        """
        self.__scaler = MinMaxScaler()
        self.scaled_numerical_data = self.__scaler.fit_transform(self.air_quality_uci_data.numerical_data)

    def __split_dataset(self, train_size: float) -> None:
        """Split dataset into one for training and one for testing.

        :param float train_size: a training dataset size
        """
        self.train_data, self.test_data = train_test_split(self.scaled_numerical_data, train_size=train_size, shuffle=False)

    def __split_all_dataset_into_x_and_y(self, time_steps: int) -> None:
        """Prepare the dataset to train.

        :param int time_steps: the length of data to train
        """
        self.time_steps = time_steps

        self.train_x, self.train_y = self.__split_dataset_into_x_and_y(self.train_data)
        self.test_x, self.test_y = self.__split_dataset_into_x_and_y(self.test_data)

    def __split_dataset_into_x_and_y(self, original_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the dataset to train.

        :param np.ndarray original_data: an original data to be inputted
        :return Tuple[np.ndarray, np.ndarray]: training data and target data
        """
        y_indices = np.arange(start=self.time_steps, stop=len(original_data), step=self.time_steps)
        ys = original_data[y_indices]
        
        n_data = len(ys)
        xs = original_data[range(self.time_steps * n_data)]
        xs = np.reshape(xs, (n_data, self.time_steps, len(xs[0]), 1))

        return xs, ys

    def build_rnn(self, n_units: Dict[str, int], activations: Dict[str, str]) -> None:
        """Build an RNN.

        :param Dict[str, int] n_units: the numbers of units on an RNN layer and a hidden layer
        :param Dict[str, str] activations: activation function names on an RNN layer and a hidden layer
        """
        len_data = len(self.train_x[0][0])
        self.__model = Sequential()
        self.model.add(Input(shape=(self.time_steps, len_data), name="Input"))
        self.model.add(SimpleRNN(units=n_units["rnn"], activation=activations["rnn"], name="RNN"))
        self.model.add(Dense(units=n_units["hidden"], activation=activations["hidden"], name="Hidden"))
        self.model.add(Dense(units=len_data, activation='linear', name="Output"))

    def train(self, optimizer: str, loss: str, metrics: List[str], batch_size: int, epochs: int, shuffle: bool) -> None:
        """Train the model.

        :param str optimizer: optimizer
        :param str loss: loss function
        :param List[str] metrics: metrics
        :param int batch_size: batch size
        :param int epochs: the number of epochs
        :param bool shuffle: whether the dataset shuffles or not
        :raises AttributeError: if self.model is None
        """
        if self.model is None:
            raise AttributeError("There is no model. First, you have to build your model.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None)

        self.model.fit(self.train_x, self.train_y, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def evaluate(self) -> None:
        """Evaluate the model.

        :raises AttributeError: if self.model is None
        """
        if self.model is None:
            raise AttributeError("There is no model. First, you have to build your model.")
        self.evaluated_data = self.model.predict(self.test_x)
        self.mse = mean_squared_error(self.test_y, self.evaluated_data)

        print(f"Mean squared error: {self.mse}.")
