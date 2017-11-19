import pandas as pd

from classification import k_means, decision_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, fowlkes_mallows_score
from utils import ceil, max_tree_height


def split_data(data):
    df = shuffle(pd.read_csv(data)).reset_index(drop=True)
    x, y = df.drop('activity', axis=1), df['activity']
    return train_test_split(x, y, test_size=0.25, random_state=36)


def dist(labels):
    labels = list(labels)
    return {label: 100*labels.count(label)/len(labels) for label in labels}


def average_scores(clf, param_grid, metrics, runs=10):
    classifiers = []
    for _ in range(runs):
        x_train, x_test, y_train, y_test = split_data('data/data.csv')

        classifier = clf(x_train, y_train)

        classifier_gs = GridSearchCV(estimator=classifier, cv=5, refit=True, param_grid=param_grid)
        classifier_gs.fit(x_train, y_train)

        classifiers.append((classifier, classifier_gs))

    accuracies = []
    for classifier, classifier_gs in classifiers:
        scores = {
            'initial': [],
            'final': []
        }
        for metric in metrics:
            scores['initial'].append(metric(y_test, classifier.predict(x_test)))
            scores['final'].append(metric(y_test, classifier_gs.predict(x_test)))

        accuracies.append(scores)

    return tuple(map(lambda x: sum(x) / len(x), accuracies.values()))


def display_accuracy(clf_string, clf, metric=accuracy_score):
    if clf is k_means:
        param_grid = {'n_init': range(1, 11)}
    else:
        param_grid = {
            'max_depth': range(1, max_tree_height(105)),
            'min_samples_split': range(2, ceil(105 / 2)),
            'min_samples_leaf': range(1, 11)
        }
    acc = average_scores(clf, param_grid, metric=metric)
    if metric is accuracy_score:
        metric_str = "Accuracy"
        scores = tuple([round(100 * x, 2) for x in acc])
        percent = "%"
    else:
        metric_str = "Fowlkes Mallows"
        scores = tuple([round(x, 2) for x in acc])
        percent = ""

    length = len(clf_string) + len(metric_str)
    indent = " "*(45 - length)
    print("{} {}:{}{}{}".format(clf_string, metric_str, indent, scores[0], percent))
    indent = " " * (33 - length)
    print("Grid Search {} {}:{}{}{}".format(clf_string, metric_str, indent, scores[1], percent))


def summarize_results():
    alg_1 = "Clustering"
    alg_2 = "Decision Tree"

    display_accuracy(alg_1, k_means)
    display_accuracy(alg_1, k_means, metric=fowlkes_mallows_score)

    print()

    display_accuracy(alg_2, decision_tree)
    display_accuracy(alg_2, decision_tree, metric=fowlkes_mallows_score)
