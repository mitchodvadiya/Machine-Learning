
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n DECISION TREE ")
print("Accuracy:", dt_accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, target_names=data.target_names))

plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, dt_pred),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=data.target_names,
    yticklabels=data.target_names
)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n RANDOM FOREST ")
print("Accuracy:", rf_accuracy * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=data.target_names))

plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, rf_pred),
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=data.target_names,
    yticklabels=data.target_names
)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy * 100, rf_accuracy * 100]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)
plt.title("Decision Tree vs Random Forest Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f'{value:.2f}%', ha='center')

plt.tight_layout()
plt.show()

print("\n MODEL COMPARISON ")
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

if rf_accuracy > dt_accuracy:
    print("Random Forest performed better.")
elif dt_accuracy > rf_accuracy:
    print("Decision Tree performed better.")
else:
    print("Both models performed equally.")
