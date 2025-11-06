import torch
from torch import nn
from concrete.ml.torch.compile import compile_torch_model
from ..interfaces import ExperimentOutput
from ..datasets.clean_conll import NERDatasetInfo, load_clean_conll_dataset
from .. import logger
from .NER_base import (
    convert_NER_dataset_into_pytorch_dataloaders,
    train_ner_pytorch_model,
    evaluate_ner_pytorch_model_clear,
    generate_batch_data,
    evaluate_ner_pytorch_model_fhe,
)


class NERModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        capit_classes,
        max_word_length,
        embedding_dim,
        window_size,
        hidden_dims,
        num_labels,
        dropout_rate,
    ) -> None:
        """
        vocab_size: size of the vocabulary
        capit_classes: number of capitalization classes
        max_word_length: max length of word lengths
        embedding_dim: dimension of the token embedding and double the dim of capit embedding
        window_size: number of tokens in sliding window
        hidden_dims: list of hidden layer sizes, e.g. [128, 64]
        num_labels: number of output labels
        """
        super().__init__()

        self.window_size = window_size
        self.embed = nn.Embedding(vocab_size + max_word_length + capit_classes, embedding_dim)
        input_dim = embedding_dim * window_size * 3

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
        embeds = self.embed(x)
        embeds_flattened = torch.flatten(embeds, start_dim=1, end_dim=-1)
        return self.mlp(embeds_flattened)


def experiment(tmp_dir: str) -> tuple[ExperimentOutput, NERDatasetInfo]:
    dset = load_clean_conll_dataset()
    timings = []

    window_size = 5
    capit_classes = 4
    max_word_length = 20
    model = NERModel(
        vocab_size=len(dset.vocab.vocab),
        capit_classes=capit_classes,
        max_word_length=max_word_length,
        embedding_dim=128,
        window_size=window_size,
        hidden_dims=[128, 128, 64],
        num_labels=5,
        dropout_rate=0.2,
    )

    train_dataloader, test_dataloader = convert_NER_dataset_into_pytorch_dataloaders(
        dset, window_size
    )

    train_ner_pytorch_model(train_dataloader, model)

    # evaluate clear
    y_pred_clear = evaluate_ner_pytorch_model_clear(model, dset, test_dataloader, timings)

    # compile the model
    logger.info("Compiling into concrete-ml model...")
    batch = generate_batch_data(dset, window_size, max_word_length, capit_classes)
    fhe_model = compile_torch_model(
        model, batch, n_bits=5, rounding_threshold_bits={"n_bits": 5, "method": "approximate"}
    )

    # evaluate fhe
    y_pred_fhe = evaluate_ner_pytorch_model_fhe(fhe_model, tmp_dir, dset, test_dataloader, timings)

    return ExperimentOutput(
        timings=timings,
        y_pred_clear=y_pred_clear,
        y_pred_fhe=y_pred_fhe,
    ), dset
