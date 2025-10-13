import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_dataset() -> tuple[npt.NDArray, npt.NDArray]:
    X, y = load_breast_cancer(return_X_y=True)
    return np.array(X).astype(np.float32), np.array(y)
