import time
import pickle
from pathlib import Path

from sklearn.linear_model import SGDClassifier as SKlearnSGDClassifier
from sklearn.metrics import accuracy_score
from concrete.ml.common.serialization.loaders import load
from concrete.ml.sklearn import SGDClassifier

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
    fhe_model = SGDClassifier(
        random_state=42,
        max_iter=50,
        fit_encrypted=True,
        parameters_range=(0.0, 1.0),
    )
    fhe_training_start = time.time()
    fhe_model.fit(X_train, y_train, fhe="execute")
    fhe_training_end = time.time()
    with open(model_file_fhe, "w") as file:
        fhe_model.dump(file)

    return TrainingExperimentResult(
        duration_in_sec_fhe=fhe_training_end - fhe_training_start,
        duration_in_sec_clear=clear_training_end - clear_training_start,
    )

def sgd_inference_native_model(X_train: list, X_test, _: list, y_test: list) -> InferenceExperimentResult:
    if not model_file_clear.is_file():
        raise Exception(f"Couldn't find model file (model_file_clear), please run the xgd_training experiment first!")
    if not model_file_fhe.is_file():
        raise Exception(f"Couldn't find model file (model_file_fhe), please run the xgd_training experiment first!")

    with open(model_file_clear, "rb") as file:
        model = pickle.load(file)
    clear_inference_start = time.time()
    y_pred_clear = model.predict(X_test)
    clear_inference_end = time.time()

    with open(model_file_fhe, "r") as file:
        fhe_model = load(file)
        fhe_model.compile(X_train) #required because FHE circuit is not dumped with the model
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
