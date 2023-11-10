import torch
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
num_samples = 50000
num_features = 2000
num_test_samples = 1000

# Assuming you have a list of PyTorch tensors 'X_list' and a tensor 'y' representing the target labels
X_list = [torch.rand(num_features) for _ in range(num_samples)]
y_tensor = torch.randint(0, 2, (num_samples,))
print(y_tensor)

# Convert PyTorch tensors to NumPy arrays
X_numpy = np.array([x.numpy() for x in X_list])
y_numpy = y_tensor.numpy()

# Create a DataFrame
import pandas as pd
df = pd.DataFrame({'feature': list(X_numpy), 'label': y_numpy})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['feature'].tolist(), df['label'], test_size=num_test_samples, random_state=42)

# Initialize the CatBoost classifier
model = CatBoostClassifier(iterations=50, depth=10, learning_rate=0.05, loss_function='Logloss', verbose=True)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
