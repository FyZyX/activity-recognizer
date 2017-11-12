import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

A = pd.read_csv('data/data.csv')

A = shuffle(A).reset_index(drop=True)
X, y = A.drop('activity', axis=1), A['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=36)

clf = KMeans(n_clusters=7).fit(X_train)
clf.labels_ = y_train
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
