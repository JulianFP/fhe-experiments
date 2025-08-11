from .datasets.sms_spam import load_sms_spam_dataset
from .datasets.synthetic import load_synthetic_dataset


def get_dataset_loaders():
    return {
        "synthetic": load_synthetic_dataset,
        "sms_spam": load_sms_spam_dataset,
    }
