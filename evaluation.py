import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def split_data(data):
    df = shuffle(pd.read_csv(data)).reset_index(drop=True)
    x, y = df.drop('activity', axis=1), df['activity']
    return train_test_split(x, y, test_size=0.25, random_state=36)


def accuracy(clf, x, y):
    return accuracy_score(y, clf.predict(x))


def dist(labels):
    labels = list(labels)
    return {label: 100*labels.count(label)/len(labels) for label in labels}


def display_accuracy(clf_string, clf, x, y, grid_search=False):
    acc = accuracy(clf, x, y)
    percent = round(100 * acc, 2)
    grid = "Grid Search " if grid_search else ""
    length = len(grid) + len(clf_string) + 16
    indent = " "*(45 - length)
    return "{}{} Accuracy:{}{}%".format(grid, clf_string, indent, percent)
