# Housing Price Prediction in Copenhagen

This project analyzes historical housing price data in Copenhagen to predict future prices based on changes over time. Using Python and rudimentary machine learning, the project preprocesses data, trains a simple linear regression model, and evaluates its performance.

---

## Features

- Load and preprocess housing price data from a CSV file.
- Convert percentage changes to numerical values for analysis.
- Train a linear regression model to predict prices per square meter.
- Visualize actual vs. predicted prices to assess model performance.
- Explain the calculation behind predicted prices using the trained regression model.

---

## How Predicted Prices Are Calculated

The predicted prices are calculated using a linear regression model. The model learns a linear relationship between the following features:

- **Percentage Change (Ændring):** Short-term price movement in percentage.
- **One-Year Change (Ændring 1 år):** Annual price movement in percentage.

The model uses the formula:
Predicted Price = Intercept + (Coefficient_1 × Ændring) + (Coefficient_2 × Ændring 1 år)

## Installation

Clone the Repository
```bash
https://github.com/WilliamBlochNielsen/Stock-Price-Analysis.git
```
Install required library
```bash
pip install pandas scikit-learn matplotlib
```
## Important Notes on Model Results
The results of the current model are subpar, as indcated by the evaluation metrics and the scatter plot of actual vs. predicted prices. The predicted values do not align closely with the actual values, suggesting that the features used may not adequately capture the variability in housing prices and the linear regression model might not be the best fit for this dataset.
