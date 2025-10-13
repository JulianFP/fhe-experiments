import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits


def load_digits_dataset() -> tuple[npt.NDArray, npt.NDArray]:
    X, y = load_digits(return_X_y=True)
    return np.array(X).astype(np.float32), np.array(y)
