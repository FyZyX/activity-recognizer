from classification import k_means, decision_tree
from evaluation import split_data, display_accuracy, dist
from visualize import generate_tree
from sklearn.model_selection import GridSearchCV
from utils import ceil, max_tree_height


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('data/data.csv')
    print(dist(y_train))
    num_samples = len(y_train) + len(y_test)

    cluster = k_means(X_train, y_train)
    tree = decision_tree(X_train, y_train)

    cluster_gs = GridSearchCV(estimator=cluster, cv=5, param_grid={'n_init': range(1, 11)})
    cluster_gs.fit(X_train, y_train)
    print(cluster_gs.best_params_)

    tree_gs = GridSearchCV(estimator=tree, cv=5, param_grid={
        'max_depth': range(1, max_tree_height(num_samples)),
        'min_samples_split': range(5, ceil(num_samples/2)),
        'min_samples_leaf': range(2, 11)
    })
    tree_gs.fit(X_train, y_train)
    print(tree_gs.best_params_)

    # visualize tree
    generate_tree(tree)

    alg_1 = "Clustering"
    alg_2 = "Decision Tree"
    print(display_accuracy(alg_1, cluster, X_test, y_test))
    print(display_accuracy(alg_2, tree, X_test, y_test))

    print(display_accuracy(alg_1, cluster_gs, X_test, y_test, grid_search=True))
    print(display_accuracy(alg_1, tree_gs, X_test, y_test, grid_search=True))
