import torch
import numpy as np
from catboost import CatBoostClassifier
from src.naive_dnn.utils import compute_eer
from sklearn.metrics import accuracy_score
# Generate synthetic data

import torch
import numpy as np
from catboost import CatBoostClassifier
from src.naive_dnn.utils import compute_eer
from sklearn.metrics import accuracy_score

# Generate synthetic data
def train(X_train, y_train, X_test, y_test):
    print("START TRAINING")

    # Initialize the CatBoost classifier
    model = CatBoostClassifier(iterations=700, depth=7, learning_rate=0.04, loss_function='Logloss', verbose=True)

    # Specify the validation dataset
    eval_dataset = (X_test, y_test)

    # Train the model and monitor performance on the test dataset
    model.fit(X_train, y_train, eval_set=eval_dataset, verbose=True)
    model.save_model("catboost_model.cbm")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")

# Example usage
# Replace X_train, y_train, X_test, and y_test with your actual data
# train(X_train, y_train, X_test, y_test)
