import numpy as np


def load_xor_split_dataset():
    X = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])

    # some models expect datasets of a certain size, so we just put the same samples in multiple times
    X = np.repeat(X, 10, 0)
    y = np.repeat(y, 10, 0)
    rng = np.random.default_rng(42)
    p1 = rng.permutation(len(X))
    p2 = rng.permutation(len(X))
    X_train, y_train = X[p1], y[p1]
    X_test, y_test = X[p2], y[p2]

    # add some noise so that train and test set are not identical and to make them floats
    noise1 = rng.uniform(-0.24, 0.24, (len(X), 2)).astype(np.float32)
    noise2 = rng.uniform(-0.24, 0.24, (len(X), 2)).astype(np.float32)
    X_train = np.add(X_train, noise1)
    X_test = np.add(X_test, noise2)

    return X_train, X_test, y_train, y_test
