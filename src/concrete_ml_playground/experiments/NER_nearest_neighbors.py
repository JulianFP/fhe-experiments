import torch
import time
import numpy as np
from transformers import BertTokenizer, BertModel
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import KNeighborsClassifier
from sklearn.decomposition import PCA

from .NER_base import print_inference_result_debug_messages
from ..interfaces import ExperimentOutput
from ..datasets.clean_conll import (
    RawNERDatasetInfo,
    load_raw_clean_conll_dataset,
    analyze_label_set,
)
from .. import logger


def generate_embeddings_with_bert(
    bert_tokenizer, bert_model, X: list[tuple[list[str], int]], pca=None
):
    X_no_tokens: list[str] = list(" ".join(sample[0]) for sample in X)
    X_result = []
    with torch.no_grad():
        last_sentence = ""
        latest_embeddings = []
        for (_, index), sentence in zip(X, X_no_tokens):
            if sentence != last_sentence:
                sentence_enc = bert_tokenizer.batch_encode_plus(
                    [sentence],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                output = bert_model(
                    sentence_enc["input_ids"], attention_mask=sentence_enc["attention_mask"]
                )
                latest_embeddings = output.last_hidden_state

            X_result.append(latest_embeddings[0][index])
            last_sentence = sentence

    if pca is None:
        pca = PCA(n_components=3)
        X_result = pca.fit_transform(X_result)
    else:
        X_result = pca.transform(X_result)
    return pca, X_result


def experiment(tmp_dir: str) -> tuple[ExperimentOutput, RawNERDatasetInfo]:
    dset = load_raw_clean_conll_dataset()
    timings = []

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_model = BertModel.from_pretrained("bert-base-cased")

    logger.info("Generating Bert embeddings...")
    X_train = dset.X_train[:13]
    y_train = dset.y_train[:13]
    pca, X_train_embeddings = generate_embeddings_with_bert(bert_tokenizer, bert_model, X_train)
    pca, X_test_embeddings = generate_embeddings_with_bert(
        bert_tokenizer, bert_model, dset.X_test, pca
    )
    logger.info("Finished generating Bert embeddings")

    model = KNeighborsClassifier(n_bits=2, n_neighbors=2)

    logger.info("Training clear&FHE KNeighborsClassifier models...")
    fhe_model, clear_model = model.fit_benchmark(X_train_embeddings, y_train)
    logger.info("Finished training KNeighborsClassifier models...")

    # Evaluate the model on the test set in clear:
    y_pred_clear = []
    timings.append(time.time())
    for X in X_test_embeddings:
        y_pred_clear.append(clear_model.predict(np.array([X]))[0])
    timings.append(time.time())
    print_inference_result_debug_messages(dset.X_test, dset.y_test_str, dset.y_test, y_pred_clear)
    analyze_label_set("predicted labels", y_pred_clear)

    # Compile the model
    logger.info("Compiling FHE KNeighborsClassifier model...")
    fhe_model.compile(X_train_embeddings)
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
    for X in X_test_embeddings:
        encrypted_data_array.append(client.quantize_encrypt_serialize(np.array([X])))
    timings.append(time.time())

    # server processes data
    logger.info("Running FHE processing...")
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
        y_pred_fhe.append(client.deserialize_decrypt_dequantize(Y_enc)[0])
    timings.append(time.time())
    print_inference_result_debug_messages(dset.X_test, dset.y_test_str, dset.y_test, y_pred_fhe)
    analyze_label_set("predicted labels", y_pred_fhe)

    return ExperimentOutput(
        timings=timings,
        y_pred_clear=y_pred_clear,
        y_pred_fhe=y_pred_fhe,
        fhe_model=fhe_model,
        clear_model=clear_model,
    ), dset
