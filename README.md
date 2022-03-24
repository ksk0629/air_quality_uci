# air_quality_uci
This repository is for the dataset called AirQualityUci.

## Environment
I checked if the codes in this repository work well **only on google colab whose python version is 3.7.12**. If someone runs those codes on a google colab notebook or the terminal, all you have to do is run `pip install mlflow`. Meanwhile, if someone runs them on another environment, it could work by running `pip install -r requirements.txt`. It is automatically extracted by mlflow. Note that, `requirements.txt` is for only cpu. It could work even in gpu environment if you change some packages to use gpu.

## QuickStart
1. Download `AirQualityUci.xlsx`. You could get the xlsx file from [Air Quality Time Series data UCI](https://www.kaggle.com/datasets/aayushkandpal/air-quality-time-series-data-uci) on Kaggle.

1. Open `notebook/csv_maker.ipynb` and run all cells.

1. Run `python ./src/run.py`

`mlruns` repository would appear. You could see the result by running `mlflow ui` on the terminal and accessing the outputted url on the browser. I would appreciate it if someone give me comments.

## A little more
We can change training configs by changing contents of `config_air_quality_uci.yaml`. The config file is as follows.
```yaml
mlflow:
  experiment_name: a name of the experiment, which is used in mlflow
  run_name: a name of the run, which is used in mlflow
dataset:
  this_type: dataset column names (If this is None, all of columns are used. list or str is acceptable.)
  train_size: the ratio of training dataset size
  time_steps: the length of data to predict next data
rnn:
  n_units:
    rnn: the number of units of the RNN
    hidden: the number of units of the hidden layer
  activations:
    rnn: an activation name of the RNN
    hidden: an activation name of the hidden layer
train:
  optimizer: an optimizer name
  loss: a loss function name
  metrics: metric names, which are given as list
  batch_size: a training batch size
  epochs: the number of epochs
  shuffle: whether a dataset shuffles or not
save:
  output_path: a path to an image outputted
```

It would give us some more insight and experience to change the configs.

If you wants to change the other configs, then you should change the codes in `src`.
