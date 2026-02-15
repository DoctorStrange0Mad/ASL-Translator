import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("asl_data.csv", header=None)

# Features = first 63 columns
X = data.iloc[:, :-1]

# Labels = last column
y = data.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "asl_model.pkl")

print("Model saved as asl_model.pkl")
