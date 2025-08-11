from sklearn.datasets import make_classification


def load_synthetic_dataset():
    return make_classification(
        n_features=30,
        n_redundant=0,
        n_informative=2,
        random_state=2,
        n_clusters_per_class=1,
        n_samples=250,
    )
