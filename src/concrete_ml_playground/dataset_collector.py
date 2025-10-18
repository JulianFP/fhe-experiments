from typing import Callable

from .datasets.sms_spam import load_sms_spam_dataset
from .datasets.synthetic import load_synthetic_dataset
from .datasets.xor import load_xor_split_dataset
from .datasets.iris import load_iris_dataset
from .datasets.breast_cancer import load_breast_cancer_dataset
from .datasets.digits import load_digits_dataset
from .datasets.clean_conll import load_clean_conll_dataset


def get_dataset_loader_entry(
    ds_loader: Callable, base_name: str, feature_size: int | None = None
) -> tuple[Callable, str]:
    if feature_size is None:
        feature_size_name = "all"
    else:
        feature_size_name = str(feature_size)

    return lambda: ds_loader(feature_size), f"{base_name}, {feature_size_name} features"


def get_dataset_loaders() -> dict[str, tuple[Callable, str]]:
    return {
        "xor": (load_xor_split_dataset, "XOR problem"),
        "synth_50": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 50),
        "synth_100": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 100),
        "synth_250": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 250),
        "synth_500": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 500),
        "synth_1000": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 1000),
        "synth_2500": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 2500),
        "synth_5000": get_dataset_loader_entry(load_synthetic_dataset, "Synthetic", 5000),
        "spam_50": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 50),
        "spam_100": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 100),
        "spam_250": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 250),
        "spam_500": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 500),
        "spam_1000": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 1000),
        "spam_2500": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 2500),
        "spam_5000": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam", 5000),
        "spam_all": get_dataset_loader_entry(load_sms_spam_dataset, "SMS Spam"),
        "iris": (load_iris_dataset, "Iris"),
        "cancer": (load_breast_cancer_dataset, "Breast Cancer"),
        "digits": (load_digits_dataset, "Digits"),
        "ner": (load_clean_conll_dataset, "NER (CleanCoNLL)"),
    }
