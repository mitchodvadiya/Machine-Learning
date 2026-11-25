# Experiment 1: Data Analysis / Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("First five rows:\n", df.head())
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDataset information:")
df.info()
print("\nStatistical summary:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df.hist(figsize=(10, 7))
plt.suptitle("Iris Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.tight_layout()
plt.show()
