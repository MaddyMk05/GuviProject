#Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('cars_data.csv')

# Data Cleaning
data.dropna(inplace=True)  # Remove missing values

# Convert categorical variables to numerical
label_encoders = {}
for column in ['Name', 'Location', 'Fuel_Type', 'Transmission']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert 'Mileage', 'Engine', and 'Power' to numeric after cleaning
data['Mileage'] = data['Mileage'].astype(str).str.replace(' kmpl', '').str.replace(' ', '').astype(float)
data['Engine'] = data['Engine'].astype(str).str.replace(' CC', '').str.replace(' ', '').astype(float)

# Handle invalid entries in 'Power'
data['Power'] = data['Power'].astype(str).str.replace(' bhp', '').str.replace(' ', '')
data['Power'].replace('null', np.nan, inplace=True)  # Replace 'null' with NaN
data['Power'] = pd.to_numeric(data['Power'], errors='coerce')  # Convert to numeric, coercing errors to NaN

# Feature Selection
X = data.drop('Price', axis=1)
y = data['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Visualization of Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
