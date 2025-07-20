from pathlib import Path
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier as SKlearnSGDClassifier
from concrete.ml.common.serialization.loaders import load
from concrete.ml.sklearn import SGDClassifier

# Create the data for classification:
print("Creating training & test set...")
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

print("Training clear model...")
model = SKlearnSGDClassifier(
    random_state=42,
    max_iter=50,
)

clear_training_start = time.time()
model.fit(X_train, Y_train)
clear_training_end = time.time()
print(f"Training on clear data took: {clear_training_end - clear_training_start} seconds")

print("Evaluating clear model...")
clear_inference_start = time.time()
y_pred_clear = model.predict(X_test)
clear_inference_end = time.time()
print(f"Inference on clear test set took: {clear_inference_end - clear_inference_start} seconds")

model_file = Path("./sgd_classifier_fhe.model")
if model_file.is_file():
    print("Loading FHE model from model file...")
    with open(model_file, "r") as file:
        fhe_model = load(file)
        fhe_model.compile(X_train) #required because FHE circuit is not dumped with the model
else:
    print("Training FHE model...")
    fhe_model = SGDClassifier(
        random_state=42,
        max_iter=50,
        fit_encrypted=True,
        parameters_range=parameters_range,
    )
    fhe_training_start = time.time()
    fhe_model.fit(X_train, Y_train, fhe="execute")
    fhe_training_end = time.time()
    print(f"Training on FHE encrypted data took: {fhe_training_end - fhe_training_start} seconds")
    with open(model_file, "w") as file:
        fhe_model.dump(file)

print("Evaluating FHE model...")
fhe_inference_start = time.time()
y_pred_fhe = model.predict(X_test)
fhe_inference_end = time.time()
print(f"Inference on FHE encrypted test set took: {fhe_inference_end - fhe_inference_start} seconds")

print(
    f"{(y_pred_fhe == y_pred_clear).sum()} examples over {len(y_pred_fhe)} "
    "have an FHE inference equal to the clear inference."
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
