import torch
import time
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from ..datasets.clean_conll import LABEL_LIST, analyze_label_set, NERDatasetInfo
from .. import logger


class NERPyTorchDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X_tokens, self.X_capits, self.X_wlengths = [], [], []
        for sample in X:
            token_idxs, cap_idxs, word_lengths = sample
            self.X_tokens.append(token_idxs)
            self.X_capits.append(cap_idxs)
            self.X_wlengths.append(word_lengths)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        tokens = torch.tensor(self.X_tokens[index], dtype=torch.long)
        capits = torch.tensor(self.X_capits[index], dtype=torch.long)
        wlengths = torch.tensor(self.X_wlengths[index], dtype=torch.long)
        label = torch.tensor(self.y[index], dtype=torch.long)
        return torch.cat([tokens, capits, wlengths], dim=-1), label


def convert_NER_dataset_into_pytorch_dataloaders(
    dset: NERDatasetInfo, window_size: int
) -> tuple[DataLoader, DataLoader]:
    def collate_fn_with_mask(batch, unk_idx, pad_idx, unk_prob):
        features, labels = zip(*batch)
        features, labels = torch.stack(features), torch.stack(labels)
        tokens = features[:, :window_size].clone()
        capits = features[:, window_size : 2 * window_size]
        wlengths = features[:, 2 * window_size :]

        mask = torch.rand_like(tokens.float()) < unk_prob
        mask &= tokens != pad_idx
        tokens = tokens.masked_fill(mask, unk_idx)

        return torch.cat([tokens, capits, wlengths], dim=1), labels

    logger.info("Loading train dataset into PyTorch DataLoader...")
    train_dataset = NERPyTorchDataset(dset.X_train, dset.y_train)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: collate_fn_with_mask(
            batch,
            dset.vocab.token_to_idx(dset.vocab.UNKNOWN_TOKEN),
            dset.vocab.token_to_idx(dset.vocab.PADDING_TOKEN),
            0.05,
        ),
    )

    logger.info("Loading test dataset into PyTorch DataLoader...")
    test_dataset = NERPyTorchDataset(dset.X_test, dset.y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # to maintain same ordering as X_test_token, y_test_str
    )

    return train_dataloader, test_dataloader


def train_ner_pytorch_model(dataloader: DataLoader, model: nn.Module):
    epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Training NER Model ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_features)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")


def print_inference_result_debug_messages(X_test_token, y_test_str, y_true, y_pred):
    for X_token, y_str, true_label, pred_label in zip(X_test_token, y_test_str, y_true, y_pred):
        if pred_label == true_label:
            logger.info(f"✅ Sentence: {X_token}, label: {true_label}/'{y_str}'")
        else:
            logger.info(
                f"❌ Sentence: {X_token}, predicted label: {pred_label}/'{LABEL_LIST[pred_label]}', correct label: {true_label}/'{y_str}'"
            )


def evaluate_ner_pytorch_model_clear(
    model: nn.Module, dset: NERDatasetInfo, dataloader: DataLoader, timings: list[float]
) -> list[int]:
    logger.info("Evaluating clear NER Model...")
    model.eval()
    y_pred_clear = []
    with torch.no_grad():
        timings.append(time.time())
        for features, _ in dataloader:
            logits = model(features)
            y_pred_clear.append(int(torch.argmax(logits, dim=1).item()))
        timings.append(time.time())
    print_inference_result_debug_messages(
        dset.X_test_token, dset.y_test_str, dset.y_test, y_pred_clear
    )
    analyze_label_set("predicted labels", y_pred_clear)

    return y_pred_clear


def generate_batch_data(
    dset: NERDatasetInfo, window_size: int, max_word_length: int, capit_classes: int
):
    tokens = torch.randint(0, len(dset.vocab.vocab), size=(100, window_size)).long()
    word_length_offset = len(dset.vocab.vocab) - 1  # since the min word length is 1
    capit_offset = word_length_offset + max_word_length
    wlengths = torch.randint(
        word_length_offset, word_length_offset + max_word_length, size=(100, window_size)
    ).long()
    capits = torch.randint(
        capit_offset, capit_offset + capit_classes, size=(100, window_size)
    ).long()
    return torch.cat([tokens, capits, wlengths], dim=-1)


def evaluate_ner_pytorch_model_fhe(
    fhe_model, tmp_dir: str, dset: NERDatasetInfo, dataloader: DataLoader, timings: list[float]
) -> list[int]:
    # dev init
    model_path = f"{tmp_dir}/model_dir"
    dev = FHEModelDev(path_dir=model_path, model=fhe_model)
    dev.save()

    # client init
    client = FHEModelClient(path_dir=model_path, key_dir=f"{tmp_dir}/fhe_keys_client")
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert (
        type(serialized_evaluation_keys) is bytes
    )  # only returns tuple if include_tfhers_key is set to True

    # server init
    server = FHEModelServer(path_dir=model_path)
    server.load()

    # pre-processing
    logger.info("Running FHE pre-processing...")
    encrypted_data_array = []
    timings.append(time.time())
    for i, (features, _) in enumerate(dataloader, start=1):
        encrypted_data_array.append(client.quantize_encrypt_serialize(features.numpy()))
        logger.info(f"Finished pre-processing feature vector {i}/{len(dataloader)}")
    timings.append(time.time())

    # server processes data
    logger.info("Running FHE inference...")
    encrypted_result_array = []
    timings.append(time.time())
    for X_enc in encrypted_data_array:
        encrypted_result_array.append(server.run(X_enc, serialized_evaluation_keys))
    timings.append(time.time())

    # post-processing
    logger.info("Running FHE post-processing...")
    y_pred_fhe = []
    timings.append(time.time())
    for Y_enc in encrypted_result_array:
        y_pred_fhe.append(np.argmax(client.deserialize_decrypt_dequantize(Y_enc)))
    timings.append(time.time())

    print_inference_result_debug_messages(
        dset.X_test_token, dset.y_test_str, dset.y_test, y_pred_fhe
    )
    analyze_label_set("predicted labels", y_pred_fhe)

    return y_pred_fhe
