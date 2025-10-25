# This file defines some common interfaces for all the experiments so that they can be called in __main__.py more easily

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel
from typing import Callable, Protocol
from concrete.ml.common.utils import FheMode

from .datasets.clean_conll import NERDatasetInfo


class ExperimentResult(BaseModel):
    accuracy_fhe: float
    accuracy_clear: float
    f1_score_fhe: float
    f1_score_clear: float
    fhe_duration_preprocessing: float
    fhe_duration_processing: float
    fhe_duration_postprocessing: float
    clear_duration: float


class ExperimentResultFinal(ExperimentResult):
    dset_name: str
    dset_name_dict: str
    exp_name: str
    exp_name_dict: str
    accuracy_fhe_stdev: float
    accuracy_clear_stdev: float
    f1_score_fhe_stdev: float
    f1_score_clear_stdev: float
    fhe_duration_preprocessing_stdev: float
    fhe_duration_processing_stdev: float
    fhe_duration_postprocessing_stdev: float
    clear_duration_stdev: float


class ClearModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class FheModel(Protocol):
    def predict(self, X: np.ndarray, fhe: FheMode) -> np.ndarray: ...


@dataclass
class ExperimentOutput:
    y_pred_clear: npt.NDArray | list[float]
    y_pred_fhe: npt.NDArray | list[float]
    timings: list[float]
    clear_model: ClearModel | None = None
    fhe_trained_model: ClearModel | None = None
    fhe_model: FheModel | None = None
    data_preparation_step: Callable | None = None


ExpFunction = Callable[[str, list, list, list], ExperimentOutput]
NerExpFunction = Callable[[str, NERDatasetInfo], ExperimentOutput]
