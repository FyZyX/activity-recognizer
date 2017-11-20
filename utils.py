from math import log2


def ceil(n):
    """
    :param n: real number
    :return: largest integer greater than n
    """
    return int(n) + 1


def max_tree_height(n):
    """
    :param n: number of nodes
    :return: height of binary tree
    """
    return int(log2(n)) + 1
