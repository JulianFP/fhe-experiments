# This file defines some common interfaces for all the experiments so that they can be called in __main__.py more easily

from typing import Callable
from dataclasses import dataclass

@dataclass
class InferenceExperimentResult:
    accuracy_fhe: float
    accuracy_clear: float
    duration_in_sec_fhe: float
    duration_in_sec_clear: float

@dataclass
class TrainingExperimentResult:
    duration_in_sec_fhe: float
    duration_in_sec_clear: float

InferenceExpFunction = Callable[[list, list, list, list], InferenceExperimentResult]
TrainingExpFunction = Callable[[list, list], TrainingExperimentResult]
