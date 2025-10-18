import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_sms_spam_dataset(max_features: None | int = None):
    dataset = load_dataset("ucirvine/sms_spam", split="train")
    dataset = dataset.with_format("numpy")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(dataset["sms"]), np.array(dataset["label"]), test_size=0.2, random_state=42
    )

    X_train = vectorizer.fit_transform(X_train)  # train vectorizer
    X_test = vectorizer.transform(
        X_test
    )  # just apply to testing set without training vectorizer on it

    X_train, X_test = X_train.toarray().astype(np.float32), X_test.toarray().astype(np.float32)
    return X_train, X_test, y_train, y_test
