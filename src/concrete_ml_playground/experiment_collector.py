from .experiments.logReg import experiment as log_reg_exp
from .experiments.xgbClassifier import experiment as xgb_exp
from .experiments.neuralNet import experiment as neural_net
from .experiments.sgdClassifier_encrypted_training import sgd_training
from .interfaces import ExpFunction


def get_inference_experiments() -> dict[str, tuple[ExpFunction, str]]:
    return {
        "log_reg": (log_reg_exp, "Logistic Regression"),
        "xgb": (xgb_exp, "XGB Tree-based Classification"),
        "neural_net": (neural_net, "Neural Network Classifier"),
    }


def get_training_experiments() -> dict[str, tuple[ExpFunction, str]]:
    return {
        "sgd_class_train": (sgd_training, "SGD Classifier Training"),
    }
