import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


def load_sms_spam_dataset(max_features: None | int = None) -> tuple[npt.NDArray, npt.NDArray]:
    dataset = load_dataset("ucirvine/sms_spam", split="train")
    dataset = dataset.with_format("numpy")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(dataset["sms"])

    return X.toarray().astype(np.float32), np.array(dataset["label"])
