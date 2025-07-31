# This file defines some common interfaces for all the experiments so that they can be called in __main__.py more easily

from typing import Callable
from dataclasses import dataclass

@dataclass
class InferenceExperimentResult:
    accuracy_fhe: float
    accuracy_clear: float
    fhe_duration_preprocessing: float
    fhe_duration_processing: float
    fhe_duration_postprocessing: float
    clear_duration: float

    def __add__(self, o):
        return InferenceExperimentResult(
            accuracy_fhe=self.accuracy_fhe+o.accuracy_fhe,
            accuracy_clear=self.accuracy_clear+o.accuracy_clear,
            fhe_duration_preprocessing=self.fhe_duration_preprocessing+o.fhe_duration_preprocessing,
            fhe_duration_processing=self.fhe_duration_processing+o.fhe_duration_processing,
            fhe_duration_postprocessing=self.fhe_duration_postprocessing+o.fhe_duration_postprocessing,
            clear_duration=self.clear_duration+o.clear_duration,
        )

    def __truediv__(self, o):
        return InferenceExperimentResult(
            accuracy_fhe=self.accuracy_fhe / o,
            accuracy_clear=self.accuracy_clear / o,
            fhe_duration_preprocessing=self.fhe_duration_preprocessing / o,
            fhe_duration_processing=self.fhe_duration_processing / o,
            fhe_duration_postprocessing=self.fhe_duration_postprocessing / o,
            clear_duration=self.clear_duration / o,
        )

@dataclass
class TrainingExperimentResult:
    duration_in_sec_fhe: float
    duration_in_sec_clear: float

    def __add__(self, o):
        return TrainingExperimentResult(
            duration_in_sec_clear=self.duration_in_sec_clear+o.duration_in_sec_clear,
            duration_in_sec_fhe=self.duration_in_sec_fhe+o.duration_in_sec_fhe,
        )

    def __truediv__(self, o):
        return TrainingExperimentResult(
            duration_in_sec_clear=self.duration_in_sec_clear / o,
            duration_in_sec_fhe=self.duration_in_sec_fhe / o,
        )

InferenceExpFunction = Callable[[list, list, list, list], InferenceExperimentResult]
TrainingExpFunction = Callable[[list, list], TrainingExperimentResult]
