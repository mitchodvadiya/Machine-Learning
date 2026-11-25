# Experiment 14: Recurrent Neural Network using LSTM
# Demonstration: predict the next value in a sine-wave sequence
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(42)
tf.random.set_seed(42)

time = np.arange(0, 100, 0.1)
series = np.sin(time)

sequence_length = 20
X, y = [], []

for i in range(len(series) - sequence_length):
    X.append(series[i:i + sequence_length])
    y.append(series[i + sequence_length])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = models.Sequential([
    layers.Input(shape=(sequence_length, 1)),
    layers.LSTM(32),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test MAE:", mae)

predictions = model.predict(X_test, verbose=0).flatten()

plt.figure(figsize=(10, 4))
plt.plot(y_test[:150], label="Actual")
plt.plot(predictions[:150], label="Predicted")
plt.title("LSTM: Actual vs Predicted Values")
plt.xlabel("Test Sample")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
