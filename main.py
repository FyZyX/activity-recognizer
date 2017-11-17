from classification import k_means, decision_tree
from evaluation import split_data, accuracy
from visualize import generate_tree
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('data/data.csv')

    cluster = k_means(X_train, y_train)
    tree = decision_tree(X_train, y_train)

    cluster_gs = GridSearchCV(estimator=cluster, param_grid={'n_init': range(1, 11)}, refit=True)
    cluster_gs.fit(X_train, y_train)
    print(cluster_gs.best_params_)

    tree_gs = GridSearchCV(estimator=tree, param_grid={
        'max_depth': range(1, 11),
        'min_samples_split': range(5, 55),
        'min_samples_leaf': range(2, 11)
    }, refit=True)
    tree_gs.fit(X_train, y_train)
    print(tree_gs.best_params_)

    # visualize tree
    generate_tree(tree)

    print(accuracy(cluster_gs, X_test, y_test))
    print(accuracy(tree_gs, X_test, y_test))
