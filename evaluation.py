import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split_data():
    df = shuffle(pd.read_csv('data/data.csv')).reset_index(drop=True)
    x, y = df.drop('activity', axis=1), df['activity']
    return train_test_split(x, y, test_size=0.25, random_state=36)


def accuracy(clf, x, y):
    return accuracy_score(y, clf.predict(x))
