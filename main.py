from classification import k_means, decision_tree
from evaluation import split_data, accuracy
from visualize import generate_tree
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('data/data.csv')

    cluster = k_means(X_train, y_train)
    tree = decision_tree(X_train, y_train)

    # visualize tree
    generate_tree(tree)

    print(accuracy(cluster, X_test, y_test))
    print(accuracy(tree, X_test, y_test))
