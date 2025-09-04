import numpy as np


def load_xor_dataset():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # some models expect datasets of a certain size, so we just put the same samples in multiple times
    X = np.repeat(X, 10, 0)
    y = np.repeat(y, 10, 0)
    rng = np.random.default_rng(42)
    p = rng.permutation(len(X))
    X, y = X[p], y[p]

    # add some noise so that train and test set are not identical and to make them floats
    noise = rng.uniform(-0.5, 0.5, (len(X), 2))
    X = np.add(X, noise)

    return X, y
