import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("attack_data_realistic.csv")

# Define correct feature columns
features = [
    'Attempts',
    'IP_Change',
    'Request_Frequency',
    'User_Agent_Diversity',
      'Key_Length',
    'Error_Code_Frequency'
]

# Shuffle and split data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = data.iloc[:10000]
test_data = data.iloc[10000:]

# Correct label column name
X_train = train_data[features]
y_train = train_data["Attack"]

X_test = test_data[features]
y_test = test_data["Attack"]

# Show label distribution
print("Label distribution in test set:")
print(y_test.value_counts())

# Apply SMOTE to balance training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Plot predicted probability distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_probs, bins=30, kde=True, color="blue")
plt.xlabel("Predicted Probability of Attack")
plt.ylabel("Number of Samples")
plt.title("Predicted Probability Distribution")
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=20, kde=True, color='purple', alpha=0.7)
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Number of Samples")
plt.title("Residuals Distribution")
plt.show()

# Save model
joblib.dump(model, "brute_force_rf.pkl")
print(" Model saved as brute_force_rf.pkl")
