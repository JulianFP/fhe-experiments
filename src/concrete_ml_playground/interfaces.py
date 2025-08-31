# This file defines some common interfaces for all the experiments so that they can be called in __main__.py more easily

import numpy as np

from dataclasses import dataclass
from typing import Callable, Protocol
from concrete.ml.common.utils import FheMode


@dataclass
class ExperimentResult:
    accuracy_fhe: float
    accuracy_clear: float
    fhe_duration_preprocessing: float
    fhe_duration_processing: float
    fhe_duration_postprocessing: float
    clear_duration: float

    def __add__(self, o):
        return ExperimentResult(
            accuracy_fhe=self.accuracy_fhe + o.accuracy_fhe,
            accuracy_clear=self.accuracy_clear + o.accuracy_clear,
            fhe_duration_preprocessing=self.fhe_duration_preprocessing
            + o.fhe_duration_preprocessing,
            fhe_duration_processing=self.fhe_duration_processing + o.fhe_duration_processing,
            fhe_duration_postprocessing=self.fhe_duration_postprocessing
            + o.fhe_duration_postprocessing,
            clear_duration=self.clear_duration + o.clear_duration,
        )

    def __truediv__(self, o):
        return ExperimentResult(
            accuracy_fhe=self.accuracy_fhe / o,
            accuracy_clear=self.accuracy_clear / o,
            fhe_duration_preprocessing=self.fhe_duration_preprocessing / o,
            fhe_duration_processing=self.fhe_duration_processing / o,
            fhe_duration_postprocessing=self.fhe_duration_postprocessing / o,
            clear_duration=self.clear_duration / o,
        )


class ClearModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class FheModel(Protocol):
    def predict(self, X: np.ndarray, fhe: FheMode) -> np.ndarray: ...


@dataclass
class DecisionBoundaryPlotData:
    clear_model: ClearModel
    fhe_trained_model: ClearModel | None = None
    fhe_model: FheModel | None = None


ExpFunction = Callable[[list, list, list, list], tuple[ExperimentResult, DecisionBoundaryPlotData]]
