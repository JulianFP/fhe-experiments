import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_dataset():
    X, y = load_digits(return_X_y=True)
    X, y = np.array(X).astype(np.float32), np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)
