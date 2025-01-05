import pandas as pd

file_path = 'copenhagen-house-price-change.csv'
housing_data = pd.read_csv(file_path)

# Convert percentage columns to numeric by removing '%' and converting to float
columns_to_convert = ['Ændring', 'Ændring 1 år', 'Ændring siden 1992']
for column in columns_to_convert:
    housing_data[column] = housing_data[column].str.replace('%', '').str.replace(',', '.').astype(float)

# Extract relevant features for modeling
processed_data = housing_data[['Pris pr. m²', 'Ændring', 'Ændring 1 år']]

# Delete comment for error handling
# print(processed_data.head())
# print(housing_data.info())
# print(housing_data.head())

from sklearn.model_selection import train_test_split

X = processed_data[['Ændring', 'Ændring 1 år']]
y = processed_data['Pris pr. m²']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

import matplotlib.pyplot as plt

# Plot of actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices (Pris pr. m²)')
plt.ylabel('Predicted Prices (Pris pr. m²)')
plt.title('Actual vs. Predicted Prices')
plt.show()
