import time
import numpy as np

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concrete.ml.sklearn.xgb import XGBClassifier
from xgboost import XGBClassifier as XGBBoostXGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from ..interfaces import DecisionBoundaryPlotData, ExperimentResult


def experiment(
    tmp_dir: str, X_train: list, X_test: list, y_train: list, y_test: list
) -> tuple[ExperimentResult, DecisionBoundaryPlotData]:
    # Instantiate the model:
    model = XGBBoostXGBClassifier()

    # Fit the model:
    pipeline = Pipeline(
        [("standard_scaler", StandardScaler()), ("pca", PCA(random_state=42)), ("model", model)]
    )
    param_grid = {
        "pca__n_components": [2, 5, 10, 15],
        "model__max_depth": [2, 3, 5],
        "model__n_estimators": [5, 10, 20],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    data_transformation_pipeline = best_pipeline[:-1]
    model = best_pipeline[-1]

    # Transform test set
    X_train_transformed = data_transformation_pipeline.transform(X_train)
    X_test_transformed = data_transformation_pipeline.transform(X_test)

    # Evaluate the model on the test set in clear:
    y_pred_clear = []
    start_clear = time.time()
    for X in X_test_transformed:
        y_pred_clear.append(model.predict([X]))
    end_clear = time.time()

    # Compile the model
    cml_model = XGBClassifier.from_sklearn_model(model, X_train_transformed, n_bits=8)
    cml_model.compile(X_train_transformed)
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
    for X in X_test_transformed:
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
            data_preparation_step=data_transformation_pipeline.transform,
        ),
    )
