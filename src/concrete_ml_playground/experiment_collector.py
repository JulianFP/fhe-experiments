from .experiments.logReg import experiment as log_reg_exp
from .experiments.xgbClassifier import experiment as xgb_exp
from .experiments.neuralNet import experiment as neural_net
from .experiments.NER_PTQ import experiment as ner_ptq_experiment
from .experiments.sgdClassifier_encrypted_training import sgd_training
from .interfaces import ExpFunction, NerExpFunction


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


def get_ner_experiments() -> dict[str, tuple[NerExpFunction, str]]:
    return {
        "ner_ptq": (ner_ptq_experiment, "NER with PTQ"),
    }
