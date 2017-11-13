from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


def k_means(x, y):
    clf = KMeans(n_clusters=7).fit(x)
    clf.labels_ = y
    return clf


def decision_tree(x, y):
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    return clf
