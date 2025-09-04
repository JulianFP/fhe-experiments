from typing import Callable
from .datasets.sms_spam import load_sms_spam_dataset
from .datasets.synthetic import load_synthetic_dataset


def get_dataset_loaders() -> dict[str, tuple[Callable, str]]:
    return {
        "synth_50": (lambda: load_synthetic_dataset(50), "Synthetic, 50 features"),
        "synth_100": (lambda: load_synthetic_dataset(100), "Synthetic, 100 features"),
        "synth_250": (lambda: load_synthetic_dataset(250), "Synthetic, 250 features"),
        "synth_500": (lambda: load_synthetic_dataset(500), "Synthetic, 500 features"),
        "synth_1000": (lambda: load_synthetic_dataset(1000), "Synthetic, 1000 features"),
        "synth_2500": (lambda: load_synthetic_dataset(2500), "Synthetic, 2500 features"),
        "synth_5000": (lambda: load_synthetic_dataset(5000), "Synthetic, 5000 features"),
        "spam_50": (lambda: load_sms_spam_dataset(50), "SMS Spam, 50 features"),
        "spam_100": (lambda: load_sms_spam_dataset(100), "SMS Spam, 100 features"),
        "spam_250": (lambda: load_sms_spam_dataset(250), "SMS Spam, 250 features"),
        "spam_500": (lambda: load_sms_spam_dataset(500), "SMS Spam, 500 features"),
        "spam_1000": (lambda: load_sms_spam_dataset(1000), "SMS Spam, 1000 features"),
        "spam_2500": (lambda: load_sms_spam_dataset(2500), "SMS Spam, 2500 features"),
        "spam_5000": (lambda: load_sms_spam_dataset(5000), "SMS Spam, 5000 features"),
        "spam_all": (load_sms_spam_dataset, "SMS Spam, all features"),
    }
