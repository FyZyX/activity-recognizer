import numpy as np
import matplotlib.pyplot as plt

from os import path
from sklearn.tree import export_graphviz


def plot(axes, axis, values, c='chartreuse'):
    """
    Convenience method for graphing the acceleration graphs
    """
    a = axes[axis]
    a.set_xlabel('time (s)')
    x = np.array(range(len(values))) / 1000
    dim = 'x' if axis == 0 else 'y' if axis == 1 else 'z'
    a.set_title('-'.join([dim, 'acceleration']))
    a.plot(x, values / 1000, c=c)


def visualize(s):
    """
    Plots x, y, and z components of acceleration
    """
    n = 3
    fig, ax = plt.subplots(1, n, sharex=True, sharey=True)
    for x, y in zip(range(n), [s.x, s.y, s.z]):
        plot(ax, x, y)

    plt.tight_layout()
    plt.show()


def generate_tree(tree, out_file='tree'):
    """
    Creates graphical representation of decision tree
    """
    activities = [
        'Working at Computer',
        'Standing Up, Walking and Going up/down stairs',
        'Standing',
        'Walking',
        'Going Up/Down Stairs',
        'Walking and Talking with Someone',
        'Talking while Standing',
    ]
    file_name = path.join('..', 'proposal', out_file)
    export_graphviz(tree, out_file="{}.dot".format(file_name), class_names=activities, rounded=True)
    # system("dot -Tpng {0}.dot -o {0}.png".format(out_file))
