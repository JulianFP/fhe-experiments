import time
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from concrete.ml.torch.compile import compile_torch_model
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from ..interfaces import ExperimentOutput
from ..datasets.clean_conll import NERDatasetInfo
from .. import logger


class NERModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        capit_classes,
        embedding_dim,
        window_size,
        hidden_dims,
        num_labels,
        dropout_rate,
    ) -> None:
        """
        vocab_size: size of the vocabulary
        capit_classes: number of capitalization classes
        embedding_dim: dimension of the token embedding and double the dim of capit embedding
        window_size: number of tokens in sliding window
        hidden_dims: list of hidden layer sizes, e.g. [128, 64]
        num_labels: number of output labels
        """
        super().__init__()

        self.window_size = window_size
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        capit_embedding_dim = embedding_dim // 2
        self.capit_embed = nn.Embedding(capit_classes, capit_embedding_dim)
        input_dim = (embedding_dim + capit_embedding_dim + 1) * window_size

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_labels))  # Output layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: feature tensor containing token indices, capitalization information and word lengths, all with length self.window_size
        """
        if x.dim() == 1:  # ensure batch dimension
            x = x.unsqueeze(0)

        old_size = x.size(0)

        token_idxs = x[:, : self.window_size]
        capits = x[:, self.window_size : 2 * self.window_size]
        wlengths = x[:, 2 * self.window_size :]

        token_embeds = self.token_embed(token_idxs)
        capit_embeds = self.capit_embed(capits)
        wlengths = wlengths.float().unsqueeze(
            -1
        )  # since the embeds are 3D, batches of lists of 5 128-dim vectors
        features = torch.cat([token_embeds, capit_embeds, wlengths], dim=-1)
        x = features.view(old_size, -1)  # flatten all features
        return self.mlp(x)


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


def print_inference_result_debug_messages(vocab, X_test_token, y_test_str, y_true, y_pred):
    for X_token, y_str, true_label, pred_label in zip(X_test_token, y_test_str, y_true, y_pred):
        if pred_label == true_label:
            logger.info(f"✅ Sentence: {X_token}, label: {true_label}/'{y_str}'")
        else:
            logger.info(
                f"❌ Sentence: {X_token}, predicted label: {pred_label}/'{vocab.label_idx_to_string(pred_label)}', correct label: {true_label}/'{y_str}'"
            )


def experiment(tmp_dir: str, dset: NERDatasetInfo) -> ExperimentOutput:
    timings = []

    window_size = 5
    capit_classes = 4
    model = NERModel(
        vocab_size=len(dset.vocab.vocab),
        capit_classes=capit_classes,
        embedding_dim=128,
        window_size=window_size,
        hidden_dims=[128, 128, 64],
        num_labels=5,
        dropout_rate=0.2,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    logger.info("Loading train dataset into PyTorch DataLoader...")
    train_dataset = NERPyTorchDataset(dset.X_train, dset.y_train)

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

    epochs = 1
    logger.info(f"Training NER Model ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_y in train_dataloader:
            optimizer.zero_grad()
            preds = model(batch_features)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_dataloader):.4f}")

    logger.info("Loading test dataset into PyTorch DataLoader...")
    test_dataset = NERPyTorchDataset(dset.X_test, dset.y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # to maintain same ordering as X_test_token, y_test_str
    )

    logger.info("Evaluating NER Model...")
    model.eval()
    y_pred_clear = []
    with torch.no_grad():
        timings.append(time.time())
        for features, _ in test_dataloader:
            logits = model(features)
            y_pred_clear.append(int(torch.argmax(logits, dim=1).item()))
        timings.append(time.time())
    print_inference_result_debug_messages(
        dset.vocab, dset.X_test_token, dset.y_test_str, dset.y_test, y_pred_clear
    )
    dset.vocab.analyze_label_set("predicted labels", y_pred_clear)

    logger.info("Compiling into concrete-ml model...")
    tokens = torch.randint(0, len(dset.vocab.vocab), size=(100, window_size)).long()
    capits = torch.randint(0, capit_classes, size=(100, window_size)).long()
    wlengths = torch.randint(0, 20, size=(100, window_size)).long()
    batch = torch.cat([tokens, capits, wlengths], dim=-1).float()
    # compile the model
    fhe_model = compile_torch_model(model, batch, n_bits=8, rounding_threshold_bits=8, p_error=0.01)
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
    for features, _ in test_dataloader:
        encrypted_data_array.append(client.quantize_encrypt_serialize(features))
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

    return ExperimentOutput(
        timings=timings,
        y_pred_clear=y_pred_clear,
        y_pred_fhe=y_pred_fhe,
    )
