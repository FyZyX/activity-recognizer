from math import log2


def ceil(n):
    return int(n) + 1


def max_tree_height(n):
    return int(log2(n)) + 1