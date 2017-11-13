from classification import k_means, decision_tree
from evaluation import split_data, accuracy

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data()()

    print(accuracy(k_means(X_train, y_train), X_test, y_test))
    print(accuracy(decision_tree(X_train, y_train), X_test, y_test))