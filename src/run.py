import argparse
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml

import air_quality_uci_dataset
import air_quality_uci_rnn


def train_and_evaluate(this_type: Optional[Union[List[str], str]], train_size: float, time_steps: int,
                       n_units: Dict[str, int], activations: Dict[str, str], optimizer: str, loss: str, metrics: List[str], batch_size: int, epochs: int, shuffle: bool) -> object:
    """Train an rnn model on air quality uci dataset and evaluate it.

    :param Optional[Union[List[str], str]] this_type: string or list of strings to define elements of this dataset, defaults to None
    :param float train_size: ratio of training dataset size
    :param int time_steps: the length of data to predict
    :param Dict[str, int] n_units:  the numbers of units on an RNN layer and a hidden layer
    :param Dict[str, str] activations: activation function names on an RNN layer and a hidden layer
    :param str optimizer: optimizer
    :param str loss: loss function
    :param List[str] metrics: metrics
    :param int batch_size: batch size
    :param int epochs: the number of epochs
    :param bool shuffle: whether the dataset shuffles or not
    :return object: trained rnn model on air quality uci dataset
    """
    # Load the air quality uci dataset
    air_quality_uci_data = air_quality_uci_dataset.AirQualityUciDataset(this_type=this_type)

    # Create an RNN for air quality uci
    model = air_quality_uci_rnn.AirQualityUciRNN()

    # Prepare to train
    model.prepare_to_train(air_quality_uci_data=air_quality_uci_data, train_size=train_size, time_steps=time_steps)

    # Build a model
    model.build_rnn(n_units=n_units, activations=activations)

    # Train the model
    model.train(optimizer=optimizer, loss=loss, metrics=metrics, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    # Evaluate the model
    model.evaluate()
    
    return model


def save_evaluation_result(trained_model: object, output_path: str) -> None:
    """Save a given information.

    :param object trained_model: a trained model that has already been evaluated
    :param str output_path: a path to an output file
    """
    evaluated_data = trained_model.evaluated_data
    test_y = trained_model.test_y

    size = len(test_y[0])
    num_columns = 4
    num_rows = int(size // num_columns + 1)
    f, axs = plt.subplots(nrows=num_rows, ncols=num_columns, sharex=True, sharey=True)
    len_data = len(test_y)
    row_index = 0
    col_index = 0
    for _ in range(size):
        pred_color = "red"
        true_color = "blue"

        axs[row_index, col_index].plot(range(len_data), evaluated_data[:, row_index + col_index], linewidth=0.5, label="prediction", color=pred_color)
        axs[row_index, col_index].plot(range(len_data), test_y[:, row_index + col_index], linewidth=0.5, label="true value", color=true_color)
        all_values = [evaluated_data[:, row_index + col_index], test_y[:, row_index + col_index]]
        min_y = np.min(all_values)
        max_y = np.max(all_values)
        axs[row_index, col_index].set_ylim([min_y - 0.1, max_y + 0.1])
        plt.xlabel(f"prediction: {pred_color}, true: {true_color}")
        if col_index != num_columns - 1:
            col_index += 1
        else:
            col_index = 0
            row_index += 1

    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RNN for air quality uci dataset.")

    parser.add_argument("-c", "--config_yaml_path", required=False, type=str, default="./config_air_quality_uci.yaml")
    args = parser.parse_args()

    # Load configs
    with open(args.config_yaml_path, "r") as yaml_f:
        config = yaml.safe_load(yaml_f)
    config_mlflow = config["mlflow"]

    # Start training and evaluating whilist logging the information
    mlflow.set_experiment(config_mlflow["experiment_name"])
    with mlflow.start_run(run_name=config_mlflow["run_name"]):
        mlflow.keras.autolog()

        config_dataset = config["dataset"]
        config_rnn = config["rnn"]
        config_train = config["train"]
        trained_model = train_and_evaluate(**config_dataset, **config_rnn, **config_train)

        config_save = config["save"]
        save_evaluation_result(trained_model=trained_model, **config_save)

        mlflow.log_artifact(args.config_yaml_path)
        mlflow.log_artifact(config_save["output_path"])
