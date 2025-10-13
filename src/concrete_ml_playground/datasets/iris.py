import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris


def load_iris_dataset() -> tuple[npt.NDArray, npt.NDArray]:
    X, y = load_iris(return_X_y=True)
    return np.array(X).astype(np.float32), np.array(y)
