import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_synthetic_dataset(features: int):
    X, y = make_classification(
        n_features=features,
        n_redundant=0,
        n_informative=features // 10,
        random_state=2,
        n_clusters_per_class=1,
        n_samples=250,
    )
    X = np.array(X).astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)
