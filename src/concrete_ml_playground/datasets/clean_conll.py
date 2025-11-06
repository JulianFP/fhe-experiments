import os
import subprocess
import random
import re
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import Counter
from .. import logger

repo_url = "https://github.com/flairNLP/CleanCoNLL"
clone_dir = "CleanCoNLL_dataset-repo"

random.seed(42)


DS_LABEL_MAP = {
    "O": 0,
    "B-LOC": 1,
    "I-LOC": 1,
    "B-ORG": 2,
    "I-ORG": 2,
    "B-PER": 3,
    "I-PER": 3,
    "B-MISC": 4,
    "I-MISC": 4,
}
LABEL_LIST = ["O", "LOC", "ORG", "PER", "MISC"]


def analyze_label_set(lset_name: str, y):
    total_count = len(y)
    logger.info(f"Total amount of samples in {lset_name}: {total_count}")
    labels_count = {clas: 0.0 for clas in DS_LABEL_MAP.values()}
    for label in y:
        labels_count[label] += 1
    for label, count in labels_count.items():
        labels_count[label] = count / total_count
    logger.info(f"Relative label occurrences in {lset_name}: {labels_count}")


def token_filter(token: str) -> bool:
    """
    To make it easier to the model, we exclude certain special tokens from NER analysis
    These tokens will be part of the sliding window, but never in the center
    (i.e. the model will never have to guess if this token is an entity or not)
    These tokens are punctuation and numbers
    """
    if re.match(".*[a-zA-Z]", token):
        return True
    else:
        return False


def convert_dataset_to_samples(
    sentences: list[list[str]], label_lists: list[list[str]]
) -> tuple[list[tuple[list[str], int]], list[str]]:
    X: list[tuple[list[str], int]] = []
    y: list[str] = []

    for sentence, label_list in zip(sentences, label_lists):
        included_indices: list[int] = []
        # include all words that are entities:
        for i, label in enumerate(label_list):
            if label != "O" and token_filter(sentence[i]):
                included_indices.append(i)

        # include at most an equal amount of non-entity words
        if 2 * len(included_indices) >= len(label_list) - 4:
            non_entity_indices = list(range(2, len(label_list) - 2))
        else:
            non_entity_indices = random.choices(
                [i for i in range(2, len(label_list) - 2) if i not in included_indices],
                k=len(included_indices),
            )
        for i in non_entity_indices:
            if token_filter(sentence[i]):
                included_indices.append(i)

        for i in included_indices:
            X.append((sentence, i))
            y.append(label_list[i])

    return X, y


def convert_dataset_labels_to_ints(labels: list[str]):
    y: list[int] = []
    for label in labels:
        y.append(DS_LABEL_MAP[label])
    return y


class Vocabulary:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"

    def __init__(self, sentences) -> None:
        word_counts = Counter(word.lower() for sent in sentences for word in sent)
        self.vocab = {
            self.UNKNOWN_TOKEN: 0,
            self.PADDING_TOKEN: 1,
            **{word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())},
        }
        logger.info(f"Generated vocabulary of size {len(self.vocab)}")

    def token_to_idx(self, token: str):
        idx = self.vocab.get(token.lower())
        if idx is None:
            idx = 0
        return idx

    def convert_samples_to_padded_window_samples(
        self, X: list[tuple[list[str], int]]
    ) -> list[list[str]]:
        X_window: list[list[str]] = []

        for sentence, index in X:
            padding = [self.PADDING_TOKEN, self.PADDING_TOKEN]
            new_sentence = padding + sentence + padding

            # offset i in idx_sentence by +2 since we padded with two tokens in beginning
            # i.e. in reality this is range [i-2:i+3]
            features = new_sentence[index : index + 5]
            X_window.append(features)

        return X_window

    def convert_token_samples_to_features(self, samples: list[list[str]]) -> npt.NDArray:
        max_word_length = 20
        word_length_offset = len(self.vocab) - 1  # since the min word length is 1
        capit_offset = word_length_offset + max_word_length
        X = []
        for sample in samples:
            token_idxs = []
            capitalizations = []
            word_lengths = []
            for word in sample:
                # capitalization
                capit = capit_offset  # default: no character is a capital letter
                if re.match("^[A-Z][A-Z-_.]+$", word):
                    # whole word (longer than 1 letter) consists of capitalized letters (or '-', '_', '.')
                    capit = capit_offset + 3
                elif re.match("^[A-Z]", word):
                    # first character is a capital letter
                    capit = capit_offset + 2
                elif re.match("[A-Z]", word):
                    # any character is a capital letter
                    capit = capit_offset + 1
                capitalizations.append(capit)

                # word length (normalized to max length of max_word_length)
                word_lengths.append(word_length_offset + min(len(word), max_word_length))

                # token idx
                token_idxs.append(self.token_to_idx(word))
            X.append([token_idxs, capitalizations, word_lengths])

        return np.array(X, dtype=np.uint32)


@dataclass
class RawNERDatasetInfo:
    X_train: list[tuple[list[str], int]]
    X_test: list[tuple[list[str], int]]
    y_train: npt.NDArray
    y_test: npt.NDArray
    y_test_str: list[str]


@dataclass
class NERDatasetInfo:
    vocab: Vocabulary
    X_train: npt.NDArray
    X_test: npt.NDArray
    X_test_token: list[list[str]]
    y_train: npt.NDArray
    y_test: npt.NDArray
    y_test_str: list[str]


def parse_conll_file(file_path) -> tuple[list[list[str]], list[list[str]]]:
    sentences: list[list[str]] = []
    label_lists: list[list[str]] = []
    with open(file_path, "r") as file:
        current_sentence: list[str] = []
        current_label_list: list[str] = []
        for line in file:
            if line.startswith("-DOCSTART-"):
                continue

            splits = line.strip().split("\t")
            if len(splits) < 5:
                if len(current_sentence) > 0:
                    sentences.append(current_sentence)
                    label_lists.append(current_label_list)
                    current_sentence = []
                    current_label_list = []
            else:
                current_sentence.append(splits[0])
                current_label_list.append(splits[-1])

    return sentences, label_lists


def load_raw_clean_conll_dataset() -> RawNERDatasetInfo:
    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    test_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.test")
    train_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.train")

    if not os.path.exists(test_file):
        script_path = os.path.join(clone_dir, "create_cleanconll_from_conll03.sh")
        os.chmod(script_path, 0o755)
        subprocess.run(["./create_cleanconll_from_conll03.sh"], check=True, cwd=clone_dir)

    # train
    train_sentences, train_label_lists = parse_conll_file(train_file)
    X_train, y_train_str = convert_dataset_to_samples(train_sentences, train_label_lists)
    y_train = convert_dataset_labels_to_ints(y_train_str)
    analyze_label_set("CleanCoNLL - train set", y_train)

    # test
    test_sentences, test_label_lists = parse_conll_file(test_file)
    X_test, y_test_str = convert_dataset_to_samples(test_sentences, test_label_lists)
    y_test = convert_dataset_labels_to_ints(y_test_str)
    # without this step the test dataset would have almost 50,000 samples which would take ages to run homomorphically
    assert len(X_test) == len(y_test) == len(y_test_str)
    X_test, y_test, y_test_str = zip(*random.choices(list(zip(X_test, y_test, y_test_str)), k=100))
    analyze_label_set("CleanCoNLL - test set", y_test)

    return RawNERDatasetInfo(
        X_train=X_train,
        X_test=list(X_test),
        y_train=np.array(y_train),
        y_test=np.array(y_test),
        y_test_str=list(y_test_str),
    )


def load_clean_conll_dataset() -> NERDatasetInfo:
    raw_dataset = load_raw_clean_conll_dataset()

    logger.info("Building vocabulary from CleanCoNLL train dataset...")
    train_sentences = list(X[0] for X in raw_dataset.X_train)
    vocab = Vocabulary(train_sentences)

    logger.info(
        "Converting CleanCoNLL train&test datasets to sliding window with vocabulary indices representation..."
    )
    X_train_token = vocab.convert_samples_to_padded_window_samples(raw_dataset.X_train)
    X_train = vocab.convert_token_samples_to_features(X_train_token)
    X_test_token = vocab.convert_samples_to_padded_window_samples(raw_dataset.X_test)
    X_test = vocab.convert_token_samples_to_features(X_test_token)

    return NERDatasetInfo(
        vocab=vocab,
        X_train=X_train,
        X_test=X_test,
        X_test_token=X_test_token,
        y_train=raw_dataset.y_train,
        y_test=raw_dataset.y_test,
        y_test_str=raw_dataset.y_test_str,
    )
