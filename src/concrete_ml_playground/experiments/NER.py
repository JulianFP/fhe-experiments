import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from ..datasets.clean_conll import Vocabulary, load_clean_conll_dataset
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

    def forward(self, token_idx, capits, wlengths):
        """
        x: tensor of token indices, shape [ba
        """
        token_embeds = self.token_embed(token_idx)  # [batch_size, window_size, embedding_dim]
        capit_embeds = self.capit_embed(capits)
        features = torch.cat([token_embeds, capit_embeds, wlengths.unsqueeze(-1)], dim=-1)
        x = features.view(features.size(0), -1)  # flatten all features
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
        return tokens, capits, wlengths, label


def experiment(vocab: Vocabulary, X_train, X_test, X_test_token, y_train, y_test, y_test_str):
    model = NERModel(
        vocab_size=len(vocab.vocab),
        capit_classes=4,
        embedding_dim=128,
        window_size=5,
        hidden_dims=[128, 128, 64],
        num_labels=5,
        dropout_rate=0.2,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    logger.info("Loading train dataset into PyTorch DataLoader...")
    train_dataset = NERPyTorchDataset(X_train, y_train)

    def collate_fn_with_mask(batch, unk_idx, pad_idx, unk_prob):
        tokens_list, caps_list, lengths_list, labels_list = zip(*batch)
        tokens = torch.stack(tokens_list)
        caps = torch.stack(caps_list)
        lengths = torch.stack(lengths_list)
        labels = torch.stack(labels_list)

        if unk_prob > 0:
            mask = torch.rand_like(tokens.float()) < unk_prob
            mask &= tokens != pad_idx
            tokens = tokens.masked_fill(mask, unk_idx)

        return tokens, caps, lengths, labels

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: collate_fn_with_mask(
            batch,
            vocab.token_to_idx(vocab.UNKNOWN_TOKEN),
            vocab.token_to_idx(vocab.PADDING_TOKEN),
            0.05,
        ),
    )

    epochs = 5
    logger.info(f"Training NER Model ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X_tokens, batch_X_capits, batch_X_wlengths, batch_y in train_dataloader:
            optimizer.zero_grad()
            preds = model(batch_X_tokens, batch_X_capits, batch_X_wlengths)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_dataloader):.4f}")

    logger.info("Loading test dataset into PyTorch DataLoader...")
    test_dataset = NERPyTorchDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # to maintain same ordering as X_test_token, y_test_str
    )

    logger.info("Evaluating NER Model...")
    pred_y = []
    model.eval()
    with torch.no_grad():
        for sample, X_token, y_str in zip(test_dataloader, X_test_token, y_test_str):
            tokens, capits, wlengths, label = sample
            label = label.item()

            logits = model(tokens, capits, wlengths)
            pred_label = int(torch.argmax(logits, dim=1).item())
            if pred_label == label:
                logger.info(f"✅ Sentence: {X_token}, label: {label}/'{y_str}'")
            else:
                logger.info(
                    f"❌ Sentence: {X_token}, predicted label: {pred_label}/'{vocab.label_idx_to_string(pred_label)}', correct label: {label}/'{y_str}'"
                )
            pred_y.append(pred_label)
    vocab.analyze_label_set("predicted labels", pred_y)

    logger.info(f"Accuracy: {accuracy_score(y_test, pred_y)}")
    logger.info(f"F1 Score: {f1_score(y_test, pred_y, average='micro')}")


experiment(*load_clean_conll_dataset())
