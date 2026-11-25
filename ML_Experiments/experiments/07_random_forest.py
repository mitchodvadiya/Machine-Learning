# Experiment 7: Random Forest Classifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
print("\nFeature Importance:\n", pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()
