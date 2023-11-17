
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
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
def train(X_train, y_train, X_test, y_test, epoch= 700):
    y_train = [int(i) for i in y_train]
    y_test = [int(i) for i in y_test]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(type(y_train))
    print(type(y_test[0]))
    print(y_train[:5])
    print(X_train[0].shape)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("START TRAINING")

    # Initialize the CatBoost classifier
    model = CatBoostClassifier(num_trees= int(epoch), depth= 8, learning_rate=0.025, loss_function='Logloss', verbose=True, l2_leaf_reg= 2, early_stopping_rounds= 150)

    # Specify the validation dataset
    eval_dataset = (X_test, y_test)

    # Train the model and monitor performance on the test dataset
    model.fit(X_train, y_train, eval_set=eval_dataset, verbose=True)
    model.save_model("catboost_model.cbm")
    print("Save model at: catboost_model.cbm")

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    prob_output = model.predict_proba(X_test)[:, 1]
    prob_train_output = model.predict_proba(X_train)[:, 1]
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    train_accurancy = accuracy_score(y_train_pred, y_train)
    print(f"Accurancy on training set: {train_accurancy}")
    print("EVAL:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"EER: {compute_eer(y_test, prob_output)}")

# Example usage
# Replace X_train, y_train, X_test, and y_test with your actual data
# train(X_train, y_train, X_test, y_test)
