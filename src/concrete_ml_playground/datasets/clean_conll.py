import os
import subprocess
import random
import numpy as np
from collections import Counter
from .. import logger

repo_url = "https://github.com/flairNLP/CleanCoNLL"
clone_dir = "CleanCoNLL_dataset-repo"

random.seed(42)


class Vocabulary:
    UNKNOWN_TOKEN = "<UNK>"
    PADDING_TOKEN = "<PAD>"
    CLASS_MAP = {
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

    def __init__(self, sentences) -> None:
        word_counts = Counter(word for sent in sentences for word in sent)
        self.vocab = {
            self.UNKNOWN_TOKEN: 0,
            self.PADDING_TOKEN: 1,
            **{word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())},
        }
        logger.info(f"Generated vocabulary of size {len(self.vocab)}")

    def token_to_idx(self, token: str):
        idx = self.vocab.get(token)
        if idx is None:
            idx = 0
        return idx

    def analyze_label_set(self, lset_name: str, y):
        total_count = len(y)
        logger.info(f"Total amount of samples in {lset_name}: {total_count}")
        classes_count = {clas: 0.0 for clas in self.CLASS_MAP.values()}
        for clas in y:
            classes_count[clas] += 1
        for clas, count in classes_count.items():
            classes_count[clas] = count / total_count
        logger.info(f"Relative class counts in {lset_name}: {classes_count}")

    def convert_dataset_to_window_samples_with_idxs_padded(
        self, sentences: list[list[str]], tag_lists: list[list[str]]
    ):
        X = []
        y = []
        padd_idx = self.token_to_idx(self.PADDING_TOKEN)

        for sentence, tag_list in zip(sentences, tag_lists):
            included_indices: list[int] = []
            # include all words that are entities:
            for i, tag in enumerate(tag_list):
                if tag != "O":
                    included_indices.append(i)

            # include at most an equal amount of non-entity words
            if 2 * len(included_indices) >= len(tag_list) - 4:
                included_indices = list(range(2, len(tag_list) - 2))
            else:
                non_entity_indices = [
                    i for i in range(2, len(tag_list) - 2) if i not in included_indices
                ]
                included_indices += random.choices(non_entity_indices, k=len(included_indices))

            idx_sentence = [padd_idx, padd_idx]
            for word in sentence:
                idx_sentence.append(self.token_to_idx(word))
            idx_sentence.append(padd_idx)
            idx_sentence.append(padd_idx)
            for i in included_indices:
                # offset i in idx_sentence by +2 since we padded with two tokens in beginning
                # i.e. in reality this is range [i-2:i+3]
                features = np.array(idx_sentence[i : i + 5], dtype=np.uint32)
                tag = self.CLASS_MAP[tag_list[i]]
                X.append(features)
                y.append(tag)

        return np.array(X, dtype=np.uint32), np.array(y, dtype=np.uint8)


def parse_conll_file(file_path) -> tuple[list[list[str]], list[list[str]]]:
    sentences: list[list[str]] = []
    tag_lists: list[list[str]] = []
    with open(file_path, "r") as file:
        current_sentence: list[str] = []
        current_tag_list: list[str] = []
        for line in file:
            if line.startswith("-DOCSTART-"):
                continue

            splits = line.strip().split("\t")
            if len(splits) < 5:
                if len(current_sentence) > 0:
                    sentences.append(current_sentence)
                    tag_lists.append(current_tag_list)
                    current_sentence = []
                    current_tag_list = []
            else:
                current_sentence.append(splits[0])
                current_tag_list.append(splits[-1])

    return sentences, tag_lists


def load_clean_conll_dataset():
    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    test_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.test")
    train_file = os.path.join(clone_dir, "data/cleanconll/cleanconll.train")

    if not os.path.exists(test_file):
        script_path = os.path.join(clone_dir, "create_cleanconll_from_conll03.sh")
        os.chmod(script_path, 0o755)
        subprocess.run(["./create_cleanconll_from_conll03.sh"], check=True, cwd=clone_dir)

    train_sentences, train_tag_lists = parse_conll_file(train_file)
    logger.info("Building vocabulary from CleanCoNLL train dataset...")
    vocab = Vocabulary(train_sentences)
    X_train, y_train = vocab.convert_dataset_to_window_samples_with_idxs_padded(
        train_sentences, train_tag_lists
    )
    vocab.analyze_label_set("CleanCoNLL - train set", y_train)

    test_sentences, test_tag_lists = parse_conll_file(test_file)
    X_test, y_test = vocab.convert_dataset_to_window_samples_with_idxs_padded(
        test_sentences, test_tag_lists
    )

    # without this step the test dataset would have almost 50,000 samples which would take ages to run homomorphically
    X_test, y_test = zip(*random.choices(list(zip(X_test, y_test)), k=1000))
    vocab.analyze_label_set("CleanCoNLL - test set", y_test)

    return vocab, X_train, np.array(X_test), y_train, np.array(y_test)
