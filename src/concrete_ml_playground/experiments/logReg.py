import time

import numpy as np
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKlearnLogisticRegression

from ..interfaces import ExperimentOutput


def experiment(tmp_dir: str, X_train: list, X_test: list, y_train: list) -> ExperimentOutput:
    timings = []

    # Instantiate the model:
    model = SKlearnLogisticRegression()

    # Fit the model:
    model.fit(X_train, y_train)

    # Evaluate the model on the test set in clear:
    y_pred_clear = []
    timings.append(time.time())
    for X in X_test:
        y_pred_clear.append(model.predict([X]))
    timings.append(time.time())

    # Compile the model
    cml_model = LogisticRegression.from_sklearn_model(model, X_train, n_bits=8)
    cml_model.compile(X_train)
    model_path = f"{tmp_dir}/model_dir"
    dev = FHEModelDev(path_dir=model_path, model=cml_model)
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
        fhe_model=cml_model,
        clear_model=model,
    )
