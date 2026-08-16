import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print("Loading extracted features...")

X = np.load("X.npy")
y = np.load("y.npy")

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", np.unique(y))


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Train Random Forest
print("\nTraining classifier...")

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

classifier.fit(X_train, y_train)


# Make predictions
y_pred = classifier.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy: {:.2f}%".format(accuracy * 100))


# Detailed results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model
joblib.dump(classifier, "illegal_logging_classifier.pkl")

print("\nTrained model saved as:")
print("illegal_logging_classifier.pkl")