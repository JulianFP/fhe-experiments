import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from concrete.ml.common.serialization.loaders import load
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.deployment.fhe_client_server import DeploymentMode
from concrete.ml.sklearn import SGDClassifier
from sklearn.linear_model import SGDClassifier as SKlearnSGDClassifier
from sklearn.metrics import accuracy_score

from .interfaces import InferenceExperimentResult, TrainingExperimentResult

model_file_fhe = Path("./sgd_classifier_fhe.model")
model_file_clear = Path("./sgd_classifier_clear.model")


def sgd_training(X_train: list, y_train: list) -> TrainingExperimentResult:
    print("Training clear model...")
    model = SKlearnSGDClassifier(
        random_state=42,
        max_iter=50,
    )

    clear_training_start = time.time()
    model.fit(X_train, y_train)
    clear_training_end = time.time()
    with open(model_file_clear, "wb") as file:
        pickle.dump(model, file)

    print("Training FHE model...")
    # generate an example dataset that has the same number of features, targets and features distribution as our train set
    # this way we can teach our model these parameters pre-FHE without giving it the actual data
    x_min, x_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
    y_min, y_max = np.min(y_train), np.max(y_train)
    batch_size = 8
    x_compile_set = np.vstack([x_min, x_max] * (batch_size // 2))
    y_compile_set = np.array([y_min, y_max] * (batch_size // 2))

    # init model and show it the compile_set
    model_path = "./model_dir"
    fhe_model = SGDClassifier(
        random_state=42,
        max_iter=50,
        fit_encrypted=True,
        parameters_range=(0.0, 1.0),
    )
    fhe_model.fit(x_compile_set, y_compile_set, fhe="disable")

    dev = FHEModelDev(path_dir=model_path, model=fhe_model)
    dev.save(mode=DeploymentMode.TRAINING)

    # client init
    client = FHEModelClient(path_dir=model_path, key_dir="/tmp/fhe_keys_client")
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert (
        type(serialized_evaluation_keys) == bytes
    )  # only returns tuple if include_tfhers_key is set to True

    # server init
    server = FHEModelServer(path_dir=model_path)
    server.load()

    # pre-processing
    encrypted_data = client.quantize_encrypt_serialize(np.array(X_train))
    assert type(encrypted_data) == bytes

    # training
    fhe_training_start = time.time()
    encrypted_result = server.run(encrypted_data, serialized_evaluation_keys)
    fhe_training_end = time.time()

    model = client.deserialize_decrypt_dequantize(encrypted_result)
    with open(model_file_fhe, "w") as file:
        fhe_model.dump(file)

    # post-processing

    return TrainingExperimentResult(
        duration_in_sec_fhe=fhe_training_end - fhe_training_start,
        duration_in_sec_clear=clear_training_end - clear_training_start,
    )


def sgd_inference_native_model(
    X_train: list, X_test, _: list, y_test: list
) -> InferenceExperimentResult:
    if not model_file_clear.is_file():
        raise Exception(
            f"Couldn't find model file (model_file_clear), please run the xgd_training experiment first!"
        )
    if not model_file_fhe.is_file():
        raise Exception(
            f"Couldn't find model file (model_file_fhe), please run the xgd_training experiment first!"
        )

    with open(model_file_clear, "rb") as file:
        model = pickle.load(file)
    clear_inference_start = time.time()
    y_pred_clear = model.predict(X_test)
    clear_inference_end = time.time()

    with open(model_file_fhe, "r") as file:
        fhe_model = load(file)
        fhe_model.compile(X_train)  # required because FHE circuit is not dumped with the model
    fhe_inference_start = time.time()
    y_pred_fhe = model.predict(X_test)
    fhe_inference_end = time.time()

    return InferenceExperimentResult(
        accuracy_fhe=accuracy_score(y_test, y_pred_fhe),
        accuracy_clear=accuracy_score(y_test, y_pred_clear),
        duration_in_sec_fhe=fhe_inference_end - fhe_inference_start,
        duration_in_sec_clear=clear_inference_end - clear_inference_start,
    )


# Output:
# Creating training & test set...
# Training clear model...
# Training on clear data took: 0.0006995201110839844 seconds
# Evaluating clear model...
# Inference on clear test set took: 9.083747863769531e-05 seconds
# Training FHE model...
# Training on FHE encrypted data took: 1575.61953663826 seconds
# Evaluating FHE model...
# Inference on FHE encrypted test set took: 0.00011277198791503906 seconds
# 100 examples over 100 have an FHE inference equal to the clear inference.
