import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X, y = np.array(X).astype(np.float32), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)
