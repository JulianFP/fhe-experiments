import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


def load_sms_spam_dataset() -> tuple[npt.NDArray, npt.NDArray]:
    dataset = load_dataset("ucirvine/sms_spam", split="train")
    dataset = dataset.with_format("numpy")

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
    X = vectorizer.fit_transform(dataset["sms"])

    return X.toarray(), np.array(dataset["label"])
