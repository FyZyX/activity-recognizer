from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


def k_means(x, y):
    """
    Builds a simple k-means classifier with 7 clusters and fits data
    :param x: feature vectors
    :param y: labels
    :return: classifier object
    """
    clf = KMeans(n_clusters=7).fit(x)
    clf.labels_ = y
    return clf


def decision_tree(x, y):
    """
    Wrapper for a simple decision tree classifier that is fit to a set of data
    :param x: feature vectors
    :param y: labels
    :return: classifier object
    """
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    return clf
