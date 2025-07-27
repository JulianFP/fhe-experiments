from .interfaces import InferenceExpFunction, TrainingExpFunction
from .logReg import logistical_regression
from .sgdClassifier_encrypted_training import sgd_training, sgd_inference_native_model

def get_inference_experiments() -> dict[str, InferenceExpFunction]:
    return {
        "logistical_regression": logistical_regression,
        "sgd_classifier_native_model": sgd_inference_native_model,
    }

def get_training_experiments() -> dict[str, TrainingExpFunction]:
    return {
        "sgd_classifier_training": sgd_training,
    }
