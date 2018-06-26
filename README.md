# Activity Classification from Accelerometer Data

## Project Description
### Dataset
The dataset for this project can be found in the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer#).
The dataset consists of 15 participants, each of whom performed 7 different activities. The activities are as follows:
1. Working at Computer
2. Standing Up, Walking and Going Up/Down Stairs
3. Standing
4. Walking
5. Going Up/Down Stairs
6. Walking and Talking with Someone
7. Talking while Standing

The data is provided a series of measuremeants of the _x_, _y_, and _z_ components of acceleration.

The extracted feature data can be found in `data/data.csv`. See the UCI link for raw files.

### Goal
Ultimatley, we would like to use the accelerometer data to determine which activity someone is performing. We would like to extract useful patterns from the data that can be used as distinguishing features for a classification algorithm.

## Contents
### `feature_extraction.py`
Reads in the initial time series data and converts it into feature vectors. Current features include
- mean (_x_, _y_, _z_)
- standard deviation (_x_, _y_, _z_)
- dominant frequency (_x_, _y_, _z_)

### `classification.py`
Creates two classifiers
- `k_means`
- `decision_tree`

### `evaluation.py`
Uses two metrics
- `accuracy_score`
- `fowlkes_mallows`

### `visualize.py`
Create all visuals including acceleration graphs and decision tree representations.

### `utils.py`
Helper functions

### `main.py`
Run the program

## Getting Started
### Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `graphviz`

### Run the Code
To see the results of this analysis, run the following from a command prompt:

`python main.py`

**NOTE:** It may take some time to complete since it is averaging the results of ten different models
