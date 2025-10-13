# The code in this experiment was heavily inspired by the following example code provided by Zama: https://github.com/zama-ai/concrete-ml/blob/release/1.9.x/docs/advanced_examples/LogisticRegressionTraining.ipynb
import time
import numpy as np

from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.deployment.fhe_client_server import DeploymentMode
from concrete.ml.sklearn import SGDClassifier
from sklearn.linear_model import SGDClassifier as SKlearnSGDClassifier

from .. import logger
from ..interfaces import ExperimentOutput


def sgd_training(tmp_dir: str, X_train: list, X_test: list, y_train: list) -> ExperimentOutput:
    timings = []

    logger.info("Training clear model...")
    model = SKlearnSGDClassifier(
        random_state=42,
        max_iter=50,
    )

    timings.append(time.time())
    model.fit(X_train, y_train)
    timings.append(time.time())

    logger.info("Evaluating clear-trained model...")
    y_pred_clear = []
    for X in X_test:
        y_pred_clear.append(model.predict([X]))

    logger.info("Training FHE model...")
    # generate an example dataset that has the same number of features, targets and features distribution as our train set
    # this way we can teach our model these parameters pre-FHE without giving it the actual data
    x_min, x_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
    y_min, y_max = np.min(y_train), np.max(y_train)
    batch_size = 8
    x_compile_set = np.vstack([x_min, x_max] * (batch_size // 2))
    y_compile_set = np.array([y_min, y_max] * (batch_size // 2))

    # init model and show it the compile_set
    model_path = f"{tmp_dir}/model_dir"
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
    client = FHEModelClient(path_dir=model_path, key_dir=f"{tmp_dir}/fhe_keys_client")
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert (
        type(serialized_evaluation_keys) is bytes
    )  # only returns tuple if include_tfhers_key is set to True

    # server init
    server = FHEModelServer(path_dir=model_path)
    server.load()

    # pre-processing
    X_batches_enc, y_batches_enc = [], []
    timings.append(time.time())
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    weights = np.zeros((1, X_train_np.shape[1], 1))
    bias = np.zeros((1, 1, 1))
    for i in range(0, X_train_np.shape[0], batch_size):
        if i + batch_size < X_train_np.shape[0]:
            batch_range = range(i, i + batch_size)
        else:
            break

        X_batch = np.expand_dims(X_train_np[batch_range, :], 0)
        y_batch = np.expand_dims(y_train_np[batch_range], (0, 2))

        X_batch_enc, y_batch_enc, _, _ = client.quantize_encrypt_serialize(
            X_batch, y_batch, None, None
        )

        X_batches_enc.append(X_batch_enc)
        y_batches_enc.append(y_batch_enc)
    _, _, weights_enc, bias_enc = client.quantize_encrypt_serialize(None, None, weights, bias)
    timings.append(time.time())

    # training
    timings.append(time.time())
    weights_enc = fhe.Value.deserialize(weights_enc)
    bias_enc = fhe.Value.deserialize(bias_enc)
    evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
    for x_batch, y_batch in zip(X_batches_enc, y_batches_enc):
        x_batch = fhe.Value.deserialize(x_batch)
        y_batch = fhe.Value.deserialize(y_batch)
        weights_enc, bias_enc = server.run(
            (x_batch, y_batch, weights_enc, bias_enc), evaluation_keys
        )
    fitted_weights_enc = weights_enc.serialize()
    fitted_bias_enc = bias_enc.serialize()
    timings.append(time.time())

    # post-processing
    timings.append(time.time())
    weights, bias = client.deserialize_decrypt_dequantize(fitted_weights_enc, fitted_bias_enc)
    timings.append(time.time())

    logger.info("Evaluating fhe-trained model...")
    fhe_model = SKlearnSGDClassifier(
        random_state=42,
        max_iter=50,
    )
    fhe_model.coef_ = weights
    fhe_model.intercept_ = bias
    y_pred_fhe = []
    for X in X_test:
        y_pred_fhe.append(model.predict([X]))

    return ExperimentOutput(
        timings=timings,
        y_pred_clear=y_pred_clear,
        y_pred_fhe=y_pred_fhe,
        clear_model=model,
        fhe_trained_model=fhe_model,
    )
