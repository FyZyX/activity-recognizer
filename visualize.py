import numpy as np
import matplotlib.pyplot as plt


def plot(axes, axis, values, c='chartreuse'):
    a = axes[axis]
    a.set_xlabel('time (s)')
    x = np.array(range(len(values))) / 1000
    dim = 'x' if axis == 0 else 'y' if axis == 1 else 'z'
    a.set_title('-'.join([dim, 'acceleration']))
    a.plot(x, values / 1000, c=c)


def visualize(s):
    n = 3
    fig, ax = plt.subplots(1, n, sharex=True, sharey=True)
    for x, y in zip(range(n), [s.x, s.y, s.z]):
        plot(ax, x, y)

    plt.tight_layout()
    plt.show()
