
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create sample training data
# 0 = No leakage class
# 1 = Leakage-risk class
# This is synthetic example data, not real pipeline data.

data = pd.DataFrame({
    "age": [
        2, 5, 10, 15, 20, 25, 3, 8, 12, 18,
        1, 6, 11, 16, 22, 28, 4, 9, 14, 24,
        7, 13, 19, 26, 30, 5, 17, 21, 2, 10
    ],

    "material": [
        "PVC", "PVC", "PVC", "Iron", "Iron",
        "Iron", "CPVC", "CPVC", "Steel", "Steel",
        "HDPE", "HDPE", "Copper", "Copper",
        "Concrete", "Concrete", "PVC", "Iron",
        "Steel", "Galvanized Iron", "CPVC", "HDPE",
        "Copper", "Concrete", "Galvanized Iron",
        "PVC", "Steel", "Iron", "HDPE", "CPVC"
    ],

    "flow_rate": [
        10, 15, 20, 35, 45, 50, 12, 18, 30, 40,
        8, 16, 25, 28, 45, 55, 14, 32, 38, 48,
        17, 27, 42, 52, 60, 19, 35, 46, 9, 22
    ],

    "pressure": [
        25, 30, 35, 55, 65, 75, 28, 32, 50, 60,
        20, 30, 45, 40, 70, 80, 30, 55, 60, 75,
        35, 45, 65, 78, 85, 32, 58, 68, 22, 38
    ],

    "leak": [
        0, 0, 0, 1, 1, 1, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1, 0, 1, 1, 1,
        0, 0, 1, 1, 1, 0, 1, 1, 0, 0
    ]
})

# Step 2: Convert material names into numerical columns
data = pd.get_dummies(
    data,
    columns=["material"],
    dtype=int
)

# Step 3: Separate input features and target
X = data.drop("leak", axis=1)
y = data["leak"]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 5: Create the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed successfully!")
print("Test accuracy on synthetic data:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# Step 8: Save the model and input column names
joblib.dump(
    {
        "model": model,
        "columns": list(X.columns)
    },
    "leakage_model.pkl"
)

print("\nModel saved successfully as leakage_model.pkl")
print("You can now run main.py")
