import time

import numpy as np
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import LogisticRegression
from sklearn.linear_model import LogisticRegression as SKlearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ..interfaces import DecisionBoundaryPlotData, ExperimentResult


def experiment(
    tmp_dir: str, X_train: list, X_test: list, y_train: list, y_test: list
) -> tuple[ExperimentResult, DecisionBoundaryPlotData]:
    # Instantiate the model:
    model = SKlearnLogisticRegression()

    # Fit the model:
    model.fit(X_train, y_train)

    # Evaluate the model on the test set in clear:
    y_pred_clear = []
    start_clear = time.time()
    for X in X_test:
        y_pred_clear.append(model.predict([X]))
    end_clear = time.time()

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
    start_fhe_pre = time.time()
    for X in X_test:
        encrypted_data_array.append(client.quantize_encrypt_serialize(np.array([X])))
    end_fhe_pre = time.time()

    # server processes data
    encrypted_result_array = []
    start_fhe_proc = time.time()
    for X_enc in encrypted_data_array:
        encrypted_result_array.append(server.run(X_enc, serialized_evaluation_keys))
    end_fhe_proc = time.time()

    # post-processing
    y_pred_fhe = []
    start_fhe_post = time.time()
    for Y_enc in encrypted_result_array:
        y_pred_fhe.append(np.argmax(client.deserialize_decrypt_dequantize(Y_enc)))
    end_fhe_post = time.time()

    return (
        ExperimentResult(
            accuracy_fhe=accuracy_score(y_test, y_pred_fhe),
            accuracy_clear=accuracy_score(y_test, y_pred_clear),
            f1_score_fhe=f1_score(y_test, y_pred_fhe),
            f1_score_clear=f1_score(y_test, y_pred_clear),
            clear_duration=end_clear - start_clear,
            fhe_duration_preprocessing=end_fhe_pre - start_fhe_pre,
            fhe_duration_processing=end_fhe_proc - start_fhe_proc,
            fhe_duration_postprocessing=end_fhe_post - start_fhe_post,
        ),
        DecisionBoundaryPlotData(
            fhe_model=cml_model,
            clear_model=model,
        ),
    )
