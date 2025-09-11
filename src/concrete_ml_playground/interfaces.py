# This file defines some common interfaces for all the experiments so that they can be called in __main__.py more easily

from dataclasses import dataclass
import numpy as np

from pydantic import BaseModel
from typing import Callable, Protocol
from concrete.ml.common.utils import FheMode


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
class DecisionBoundaryPlotData:
    clear_model: ClearModel
    fhe_trained_model: ClearModel | None = None
    fhe_model: FheModel | None = None
    data_preparation_step: Callable | None = None


ExpFunction = Callable[
    [str, list, list, list, list], tuple[ExperimentResult, DecisionBoundaryPlotData]
]
