# Experiment 3: Multiple Variable Linear Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example: predict house price from area, bedrooms and age
data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "age": [10, 8, 5, 7, 4, 6, 3, 2, 1, 2],
    "price": [30, 35, 48, 52, 60, 65, 75, 82, 90, 96]
}
df = pd.DataFrame(data)

X = df[["area", "bedrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", predictions)
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 score:", r2_score(y_test, predictions))

new_house = pd.DataFrame({"area": [1700], "bedrooms": [3], "age": [4]})
print("Predicted price for new house:", model.predict(new_house)[0], "lakhs")
