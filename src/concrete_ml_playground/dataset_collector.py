from .datasets.sms_spam import load_sms_spam_dataset
from .datasets.synthetic import load_synthetic_dataset


def get_dataset_loaders():
    return {
        "synthetic": load_synthetic_dataset,
        "sms_spam_all_features": load_sms_spam_dataset,
        "sms_spam_5000_features": lambda: load_sms_spam_dataset(500),
        "sms_spam_2500_features": lambda: load_sms_spam_dataset(500),
        "sms_spam_1000_features": lambda: load_sms_spam_dataset(500),
        "sms_spam_500_features": lambda: load_sms_spam_dataset(500),
        "sms_spam_250_features": lambda: load_sms_spam_dataset(250),
        "sms_spam_100_features": lambda: load_sms_spam_dataset(100),
        "sms_spam_50_features": lambda: load_sms_spam_dataset(50),
    }
