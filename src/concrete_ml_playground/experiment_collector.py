from .experiments.logReg import logistic_regression
from .experiments.sgdClassifier_encrypted_training import sgd_training
from .interfaces import ExpFunction


def get_inference_experiments() -> dict[str, tuple[ExpFunction, str]]:
    return {
        "log_reg": (logistic_regression, "Logistic Regression"),
    }


def get_training_experiments() -> dict[str, tuple[ExpFunction, str]]:
    return {
        "sgd_class_train": (sgd_training, "SGD Classifier Training"),
    }
