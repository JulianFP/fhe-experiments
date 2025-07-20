import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import SGDClassifier
from sklearn.linear_model import SGDClassifier as SKlearnSGDClassifier

# Create the data for classification:
parameters_range = (0.0, 1.0)
X, y = make_classification(
    n_features=30,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
    n_samples=250,
)

# Retrieve train and test sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = SKlearnSGDClassifier(
    random_state=42,
    max_iter=50,
)

clear_training_start = time.time()
model.fit(X_train, Y_train)
clear_training_end = time.time()

clear_inference_start = time.time()
y_pred_clear = model.fit(X_train, Y_train)
clear_inference_end = time.time()

fhe_model = SGDClassifier(
    random_state=42,
    max_iter=50,
    fit_encrypted=True,
    parameters_range=parameters_range,
)

fhe_training_start = time.time()
fhe_model.fit(X_train, Y_train, fhe="execute")
fhe_training_end = time.time()

fhe_inference_start = time.time()
y_pred_fhe = model.predict(X_test)
fhe_inference_end = time.time()

print(
    f"{(y_pred_fhe == y_pred_clear).sum()} examples over {len(y_pred_fhe)} "
    "have an FHE inference equal to the clear inference."
)
print(f"Training on clear data took: {clear_training_end - clear_training_start} seconds")
print(f"Training on FHE encrypted data took: {fhe_training_end - fhe_training_start} seconds")
print(f"Inference on clear test set took: {clear_inference_end - clear_inference_start} seconds")
print(f"Inference on FHE encrypted test set took: {fhe_inference_end - fhe_inference_start} seconds")
