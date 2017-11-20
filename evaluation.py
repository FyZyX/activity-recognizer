import pandas as pd
from functools import reduce

from classification import k_means, decision_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, fowlkes_mallows_score
from utils import ceil, max_tree_height


def split_data(data):
    """
    Split the data into training and testing sets
    :param data: path to data file
    :return: 4-tuple (x_train, x_test, y_train, y_test)
    """
    df = shuffle(pd.read_csv(data)).reset_index(drop=True)
    # break the data into feature vectors and labels
    x, y = df.drop('activity', axis=1), df['activity']
    return train_test_split(x, y, test_size=0.25, random_state=36)


def dist(labels):
    """
    percentage of each class in the distribution
    :param labels: list of classes corresponding to a list of data points
    :return: dictionary of percentages
    """
    labels = list(labels)
    return {label: 100*labels.count(label)/len(labels) for label in labels}


def average_scores(clf, param_grid, metrics, runs=10):
    """
    Returns the scores for the initial and optimized version of an algorithm
    evaluated with a series of metrics
    :param clf: classifier
    :param param_grid: set of parameter to search
    :param metrics: list of evaluation metrics
    :param runs: number of times to run
    :return: tuple of accuracies
    """
    classifiers = []
    # run specified number of times
    for _ in range(runs):
        x_train, x_test, y_train, y_test = split_data('data/data.csv')

        # initial model
        classifier = clf(x_train, y_train)

        # optimized model
        classifier_gs = GridSearchCV(estimator=classifier, cv=5, refit=True, param_grid=param_grid)
        classifier_gs.fit(x_train, y_train)

        classifiers.append((classifier, classifier_gs))

    def predict(c, m):
        return m(y_test, c.predict(x_test))

    # perform evaluations for all metrics and classifiers
    scores = [[(predict(x, metric), predict(x_gs, metric)) for x, x_gs in classifiers] for metric in metrics]

    # average the scores form each classifier
    reduced_scores = [reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), score) for score in scores]
    return [(x[0][0]/x[1], x[0][1]/x[1]) for x in zip(reduced_scores, map(len, scores))]


def display_accuracy(clf_string, clf):
    """
    Convenience method for printing results
    :param clf_string: name of classifier
    :param clf: classify
    :return: prints results to standard output
    """
    if clf is k_means:
        param_grid = {'n_init': range(1, 11)}
    else:
        param_grid = {
            'max_depth': range(1, max_tree_height(105)),
            'min_samples_split': range(2, ceil(105 / 2)),
            'min_samples_leaf': range(1, 11)
        }

    scores = average_scores(clf, param_grid, metrics=[accuracy_score, fowlkes_mallows_score])
    scores[0] = [round(100 * score, 2) for score in scores[0]]
    scores[1] = [round(score, 2) for score in scores[1]]

    # keep lines the same length
    length = len(clf_string)
    indent = " "*(45 - length)
    # initial accuracy
    print("{} Accuracy:{}{}%".format(clf_string, indent, scores[0][0]))
    indent = " " * (33 - length)
    # optimized accuracy
    print("Grid Search {} Accuracy:{}{}%".format(clf_string, indent, scores[0][1]))
    indent = " " * (38 - length)
    # initial fowlkes_mallows_score
    print("{} Fowlkes Mallows:{}{}".format(clf_string, indent, scores[1][0]))
    indent = " " * (26 - length)
    # optimized fowlkes_mallows_score
    print("Grid Search {} Fowlkes Mallows:{}{}".format(clf_string, indent, scores[1][1]))


def summarize_results():
    """
    Convenience method to run both algorithms
    :return: prints results to standard output
    """
    for alg, clf in zip(["Clustering", "Decision Tree"], [k_means, decision_tree]):
        display_accuracy(alg, clf)
        print()
        display_accuracy(alg, clf)
