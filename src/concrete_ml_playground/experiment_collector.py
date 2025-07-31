from .interfaces import ExpFunction
from .logReg import logistical_regression
from .sgdClassifier_encrypted_training import sgd_training


def get_inference_experiments() -> dict[str, ExpFunction]:
    return {
        "logistical_regression": logistical_regression,
    }


def get_training_experiments() -> dict[str, ExpFunction]:
    return {
        "sgd_classifier_training": sgd_training,
    }
