import numpy as np

from sklearn.datasets import make_classification


def load_synthetic_dataset(features: int):
    X, y = make_classification(
        n_features=features,
        n_redundant=0,
        n_informative=features // 10,
        random_state=2,
        n_clusters_per_class=1,
        n_samples=250,
    )
    return np.array(X).astype(np.float32), y
