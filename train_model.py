import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ============================================================
# FUNCTION TO TRAIN ONE CLASSIFIER
# ============================================================

def train_classifier(X_file, y_file, model_file, classifier_name):

    print("\n========================================")
    print("TRAINING:", classifier_name)
    print("========================================")

    # --------------------------------------------------------
    # Load features and labels
    # --------------------------------------------------------

    X = np.load(X_file)
    y = np.load(y_file)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Classes:", np.unique(y))

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    classifier.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = classifier.predict(X_test)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n----------------------------------------")
    print("RESULTS:", classifier_name)
    print("----------------------------------------")

    print(
        "Accuracy: {:.2f}%".format(
            accuracy * 100
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        classifier,
        model_file
    )

    print("\nModel saved as:")
    print(model_file)


# ============================================================
# 1. MAIN CLASSIFIER
# ============================================================

train_classifier(
    "main_X.npy",
    "main_y.npy",
    "main_classifier.pkl",
    "MAIN CLASSIFIER"
)


# ============================================================
# 2. BIRD CLASSIFIER
# ============================================================

train_classifier(
    "bird_X.npy",
    "bird_y.npy",
    "bird_classifier.pkl",
    "BIRD CLASSIFIER"
)


# ============================================================
# 3. ANIMAL CLASSIFIER
# ============================================================

train_classifier(
    "animal_X.npy",
    "animal_y.npy",
    "animal_classifier.pkl",
    "ANIMAL CLASSIFIER"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("ALL CLASSIFIERS TRAINED SUCCESSFULLY")
print("========================================")

print("\nCreated models:")

print("main_classifier.pkl")
print("bird_classifier.pkl")
print("animal_classifier.pkl")

print("\nTraining complete!")