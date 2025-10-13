import time
import numpy as np

from concrete.ml.sklearn import NeuralNetClassifier
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
import torch

from .. import logger
from ..interfaces import ExperimentOutput


def experiment(tmp_dir: str, X_train: list, X_test: list, y_train: list) -> ExperimentOutput:
    timings = []

    params = {
        "module__n_layers": 3,
        "module__activation_function": torch.nn.ReLU,
        "max_epochs": 100,
        "verbose": True,
        "lr": 0.01,
    }
    model = NeuralNetClassifier(**params)

    logger.info("Training clear&FHE NN models...")
    # the fit_benchmark function trains both an quantized FHE model with QAT as well as an equivalent float clear model
    # the clear model uses skorch's NeuralNetClassifier with a pytorch sequential module (torch.nn.Sequential) defined in lines 1221-1256 in base.py in concrete-ml
    # it seems to consist of linear layers (torch.nn.Linear) and activation functions
    fhe_model, clear_model = model.fit_benchmark(X_train, y_train)
    logger.info("Finished training NN models")

    # Evaluate the model on the test set in clear:
    y_pred_clear = []
    timings.append(time.time())
    for X in X_test:
        y_pred_clear.append(clear_model.predict(np.array([X])))
    timings.append(time.time())

    # Compile the model
    fhe_model.compile(X_train)
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
    encrypted_data_array = []
    timings.append(time.time())
    for X in X_test:
        encrypted_data_array.append(client.quantize_encrypt_serialize(np.array([X])))
    timings.append(time.time())

    # server processes data
    encrypted_result_array = []
    timings.append(time.time())
    for X_enc in encrypted_data_array:
        encrypted_result_array.append(server.run(X_enc, serialized_evaluation_keys))
    timings.append(time.time())

    # post-processing
    y_pred_fhe = []
    timings.append(time.time())
    for Y_enc in encrypted_result_array:
        y_pred_fhe.append(np.argmax(client.deserialize_decrypt_dequantize(Y_enc)))
    timings.append(time.time())

    return ExperimentOutput(
        timings=timings,
        y_pred_clear=y_pred_clear,
        y_pred_fhe=y_pred_fhe,
        fhe_model=fhe_model,
        clear_model=clear_model,
    )
