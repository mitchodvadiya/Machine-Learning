# Housing Prices Regression

## Description

This project is a simple machine learning implementation for predicting real estate prices using Linear Regression.

The model uses property-related features such as area, number of bedrooms, number of bathrooms, year built, garage size, and distance from the city center.

The project was implemented using Python in a Jupyter Notebook through VS Code.

## Dataset

The dataset used is `real_estate_dataset.csv`.

### Input Features

* Square_Feet
* Num_Bedrooms
* Num_Bathrooms
* Year_Built
* Garage_Size
* Distance_to_Center

### Target

* Price

The `ID` column was removed before training because it is only used as an identifier.

## Approach

The following steps were performed:

1. Loaded the dataset using Pandas.
2. Removed the `ID` column.
3. Selected the required features and target variable.
4. Split the data into 80% training and 20% testing sets.
5. Trained a Linear Regression model.
6. Predicted prices for the test data.
7. Evaluated the model using MSE, RMSE, and R² Score.
8. Compared the absolute values of the regression coefficients.

## Results

The model gave the following results on the test data:

| Metric   |            Value |
| -------- | ---------------: |
| MSE      | 1,935,198,929.42 |
| RMSE     |        43,990.90 |
| R² Score |           0.8713 |

The R² score of **0.8713** means that the model explains around **87.13%** of the variation in the property prices in the test set.

### Coefficients

| Feature            | Coefficient |
| ------------------ | ----------: |
| Square_Feet        |     1008.41 |
| Num_Bedrooms       |    50566.19 |
| Num_Bathrooms      |    27437.60 |
| Year_Built         |     1481.55 |
| Garage_Size        |      964.10 |
| Distance_to_Center |    -1014.86 |

The feature with the largest absolute coefficient is **Num_Bedrooms**, with a coefficient of **50566.19**.

## Tools Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook
* VS Code

## Project Files

```text
real_estate_dataset.csv
Housing_Prices_Regression.ipynb
README.md
```

## How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn
```

Open `Housing_Prices_Regression.ipynb` in VS Code and run the cells in order.

## Conclusion

Linear Regression was used to predict property prices from the selected features. The model achieved an R² score of **0.8713** on the test data. From the coefficient analysis, **Num_Bedrooms** had the largest absolute coefficient in the model.
