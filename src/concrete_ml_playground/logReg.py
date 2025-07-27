import numpy as np
import time
from sklearn.linear_model import LogisticRegression as SKlearnLogisticRegression
from sklearn.metrics import accuracy_score

from concrete.ml.sklearn import LogisticRegression

from .interfaces import InferenceExperimentResult

def logistical_regression(X_train: list, X_test: list, y_train: list, y_test: list) -> InferenceExperimentResult:
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
    fhe_circuit = cml_model.compile(X_train)

    # prepare input data for FHE
    X_test_quantized = cml_model.quantize_input(np.array(X_test))
    fhe_circuit.keygen(force=True)
    X_test_encrypted = []
    for q_input in X_test_quantized:
        X_test_encrypted.append(fhe_circuit.encrypt([q_input]))

    # Perform the inference in FHE
    y_pred_fhe = []
    start_fhe = time.time()
    for enc_input in X_test_encrypted:
        y_pred_fhe.append(fhe_circuit.run(enc_input))
    end_fhe = time.time()

    # post processing
    y_pred_fhe_decrypted = []
    for y in y_pred_fhe:
        y_pred_fhe_decrypted.append(fhe_circuit.decrypt(y))
    y_pred_fhe_dequantized = cml_model.dequantize_output(np.array(y_pred_fhe_decrypted))
    y_pred_fhe_done = np.argmax(cml_model.post_processing(y_pred_fhe_dequantized), axis=1)

    return InferenceExperimentResult(
        accuracy_fhe=accuracy_score(y_test, y_pred_fhe_done),
        accuracy_clear=accuracy_score(y_test, y_pred_clear),
        duration_in_sec_fhe=end_fhe - start_fhe,
        duration_in_sec_clear=end_clear - start_clear,
    )

# Output:
# 100 examples over 100 have an FHE inference equal to the clear inference.
# Clear evaluation on test set took: 0.00019240379333496094 seconds
# FHE evaluation on test set took: 0.9436783790588379 seconds
