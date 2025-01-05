# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the dataset
# data = pd.read_csv('copenhagen-house-price-change.csv')

# # Display the first few rows of the dataset
# print(data.head())

# # Check for missing values and data types
# print(data.info())
# print(data.describe())

# # Remove percentage signs and convert to numeric if needed
# data['Ændring'] = data['Ændring'].str.replace('%', '').astype(float)
# data['Ændring 1 år'] = data['Ændring 1 år'].str.replace('%', '').astype(float)
# data['Ændring siden 1992'] = data['Ændring siden 1992'].str.replace('%', '').astype(float)
# data['Pris pr. m2'] = data['Pris pr. m2'].astype(float)


# # Create a "Year" and "Quarter" feature from the "Periode" column if needed
# data['Year'] = data['Periode'].str.extract(r'(\d{4})').astype(int)
# data['Quarter'] = data['Periode'].str.extract(r'(\d)\.').astype(int)


# # Define predictors and target
# X = data[['Ændring', 'Ændring 1 år', 'Ændring siden 1992']]
# y = data['Pris pr. m2']

# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd

# Load the dataset
file_path = 'copenhagen-house-price-change.csv'
housing_data = pd.read_csv(file_path)

# Display the first few rows and basic info
print(housing_data.info())
print(housing_data.head())

# Convert percentage columns to numeric by removing '%' and converting to float
columns_to_convert = ['Ændring', 'Ændring 1 år', 'Ændring siden 1992']
for column in columns_to_convert:
    housing_data[column] = housing_data[column].str.replace('%', '').str.replace(',', '.').astype(float)

# Extract relevant features for modeling
processed_data = housing_data[['Pris pr. m²', 'Ændring', 'Ændring 1 år']]

# Display the cleaned data
print(processed_data.head())

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = processed_data[['Ændring', 'Ændring 1 år']]
y = processed_data['Pris pr. m²']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

import matplotlib.pyplot as plt

# Plot actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices (Pris pr. m²)')
plt.ylabel('Predicted Prices (Pris pr. m²)')
plt.title('Actual vs. Predicted Prices')
plt.show()
