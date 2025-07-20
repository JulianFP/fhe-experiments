import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SKlearnLogisticRegression

from concrete.ml.sklearn import LogisticRegression

# Create the data for classification:
X, y = make_classification(
    n_features=30,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_clusters_per_class=1,
    n_samples=250,
)

# Retrieve train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the model:
model = SKlearnLogisticRegression()

# Fit the model:
model.fit(X_train, y_train)

# Evaluate the model on the test set in clear:
start_clear = time.time()
y_pred_clear = model.predict(X_test)
end_clear = time.time()

# Compile the model:
cml_model = LogisticRegression.from_sklearn_model(model, X_train, n_bits=8)
cml_model.compile(X_train)

# Perform the inference in FHE:
start_fhe = time.time()
y_pred_fhe = cml_model.predict(X_test, fhe="execute")
end_fhe = time.time()

# Assert that FHE predictions are the same as the clear predictions:
print(
    f"{(y_pred_fhe == y_pred_clear).sum()} examples over {len(y_pred_fhe)} "
    "have an FHE inference equal to the clear inference."
)
print(f"Clear evaluation on test set took: {end_clear-start_clear} seconds")
print(f"FHE evaluation on test set took: {end_fhe-start_fhe} seconds")

# Output:
# 100 examples over 100 have an FHE inference equal to the clear inference.
# Clear evaluation on test set took: 0.00019240379333496094 seconds
# FHE evaluation on test set took: 0.9436783790588379 seconds
