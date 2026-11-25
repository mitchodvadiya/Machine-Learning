# Experiment 2: Single Variable Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Example: predict exam marks from study hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70, 75, 85, 90, 95])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted marks:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

plt.scatter(X, y, color="blue", label="Actual data")
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(x_line, model.predict(x_line), color="red", label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Single Variable Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()
