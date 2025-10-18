import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from ..datasets.clean_conll import Vocabulary, load_clean_conll_dataset
from .. import logger


class NERModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        window_size,
        hidden_dims,
        num_classes,
        dropout_rate,
    ) -> None:
        """
        num_categories: number of unique categorical values (for the embedding)
        embedding_dim: dimension of each token embedding
        window_size: number of tokens in sliding window
        hidden_dims: list of hidden layer sizes, e.g. [128, 64]
        num_classes: number of output classes
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = embedding_dim * window_size

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))  # Output layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: tensor of token indices, shape [ba
        """
        embeds = self.embedding(x)  # [batch_size, window_size, embedding_dim]
        x = embeds.view(embeds.size(0), -1)  # flatten all embeddings
        return self.mlp(x)


def experiment(vocab: Vocabulary, X_train, X_test, y_train, y_test):
    X_train_ten, y_train_ten = torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long()

    def collate_fn_with_mask(batch, unk_idx, pad_idx, unk_prob):
        X, y = zip(*batch)
        X = torch.stack(X)
        y = torch.stack(y)

        if unk_prob > 0:
            mask = torch.rand_like(X.float()) < unk_prob
            mask &= X != pad_idx
            X = X.masked_fill(mask, unk_idx)

        return X, y

    model = NERModel(
        vocab_size=len(vocab.vocab),
        embedding_dim=128,
        window_size=5,
        hidden_dims=[128, 128, 64],
        num_classes=5,
        dropout_rate=0.2,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(X_train_ten, y_train_ten)
    logger.info("Loading dataset into PyTorch DataLoader...")
    dataloader = DataLoader(
        dataset,
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

    epochs = 4

    logger.info(f"Training NER Model ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)  # [batch, num_classes]
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    logger.info("Evaluating NER Model...")
    pred_y = []
    model.eval()
    with torch.no_grad():
        for X in X_test:
            X = torch.from_numpy(X).long().unsqueeze(0)
            logits = model(X)
            pred_class = int(torch.argmax(logits, dim=1).item())
            pred_y.append(pred_class)
    vocab.analyze_label_set("predicted labels", pred_y)

    logger.info(f"Accuracy: {accuracy_score(y_test, pred_y)}")
    logger.info(f"F1 Score: {f1_score(y_test, pred_y, average='micro')}")


experiment(*load_clean_conll_dataset())
