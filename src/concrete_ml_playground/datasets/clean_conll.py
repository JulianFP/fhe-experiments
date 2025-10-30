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


class Vocabulary:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"
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

    def label_to_idx(self, label: str) -> int:
        return self.DS_LABEL_MAP[label]

    def label_idx_to_string(self, label_id: int) -> str:
        return self.LABEL_LIST[label_id]

    def analyze_label_set(self, lset_name: str, y):
        total_count = len(y)
        logger.info(f"Total amount of samples in {lset_name}: {total_count}")
        labels_count = {clas: 0.0 for clas in self.DS_LABEL_MAP.values()}
        for label in y:
            labels_count[label] += 1
        for label, count in labels_count.items():
            labels_count[label] = count / total_count
        logger.info(f"Relative label occurrences in {lset_name}: {labels_count}")

    def token_filter(self, token: str) -> bool:
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

    def convert_dataset_to_padded_window_samples(
        self, sentences: list[list[str]], label_lists: list[list[str]]
    ) -> tuple[list[list[str]], list[str]]:
        X = []
        y = []

        for sentence, label_list in zip(sentences, label_lists):
            included_indices: list[int] = []
            # include all words that are entities:
            for i, label in enumerate(label_list):
                if label != "O" and self.token_filter(sentence[i]):
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
                if self.token_filter(sentence[i]):
                    included_indices.append(i)

            padding = [self.PADDING_TOKEN, self.PADDING_TOKEN]
            new_sentence = padding + sentence + padding
            for i in included_indices:
                # offset i in idx_sentence by +2 since we padded with two tokens in beginning
                # i.e. in reality this is range [i-2:i+3]
                features = new_sentence[i : i + 5]
                label = label_list[i]
                X.append(features)
                y.append(label)

        return X, y

    def convert_token_samples_to_features(self, samples: list[list[str]], label_lists: list[str]):
        max_word_length = 20
        word_length_offset = len(self.vocab) - 1  # since the min word length is 1
        capit_offset = word_length_offset + max_word_length
        X = []
        y = []
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
        for label in label_lists:
            y.append(self.DS_LABEL_MAP[label])

        return np.array(X, dtype=np.uint32), np.array(y, dtype=np.uint8)


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


def load_clean_conll_dataset():
    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    test_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.test")
    train_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.train")

    if not os.path.exists(test_file):
        script_path = os.path.join(clone_dir, "create_cleanconll_from_conll03.sh")
        os.chmod(script_path, 0o755)
        subprocess.run(["./create_cleanconll_from_conll03.sh"], check=True, cwd=clone_dir)

    train_sentences, train_label_lists = parse_conll_file(train_file)
    logger.info("Building vocabulary from CleanCoNLL train dataset...")
    vocab = Vocabulary(train_sentences)
    X_train_token, y_train_str = vocab.convert_dataset_to_padded_window_samples(
        train_sentences, train_label_lists
    )
    X_train, y_train = vocab.convert_token_samples_to_features(X_train_token, y_train_str)
    vocab.analyze_label_set("CleanCoNLL - train set", y_train)

    test_sentences, test_label_lists = parse_conll_file(test_file)
    X_test_token, y_test_str = vocab.convert_dataset_to_padded_window_samples(
        test_sentences, test_label_lists
    )
    X_test, y_test = vocab.convert_token_samples_to_features(X_test_token, y_test_str)

    # without this step the test dataset would have almost 50,000 samples which would take ages to run homomorphically
    assert len(X_test) == len(X_test_token) == len(y_test) == len(y_test_str)
    X_test, X_test_token, y_test, y_test_str = zip(
        *random.choices(list(zip(X_test, X_test_token, y_test, y_test_str)), k=100)
    )
    vocab.analyze_label_set("CleanCoNLL - test set", y_test)

    return NERDatasetInfo(
        vocab=vocab,
        X_train=X_train,
        X_test=np.array(X_test),
        X_test_token=list(X_test_token),
        y_train=y_train,
        y_test=np.array(y_test),
        y_test_str=list(y_test_str),
    )
