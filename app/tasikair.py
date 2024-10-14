import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Importing data from CSV file
df = pd.read_csv('file_name')

# Displaying the imported data
print("Imported Data:")
print(df)

# Defining independent (X) and dependent (y) variables
X = df['Total_Customer']  # Total customers as the independent variable
y = df['Volume_Water_m3']  # Volume of water as the dependent variable

# Adding a constant to the model
X = sm.add_constant(X)

# Building the regression model
model = sm.OLS(y, X).fit()

# Viewing the model summary
print("\nRegression Model Summary:")
print(model.summary())

# Making predictions
predictions = model.predict(X)

# Displaying prediction results
df['Predicted_Volume_of_Water'] = predictions

# Visualizing the results
plt.figure(figsize=(10, 6))
plt.scatter(df['Total_Customer'], df['Volume_Water_m3'], color='blue', label='Actual Data')
plt.plot(df['Total_Customer'], df['Predicted_Volume_of_Water'], color='red', label='Predicted', linewidth=2)
plt.title('Predicted Water Volume Based on Total Customers')
plt.xlabel('Total Customers')
plt.ylabel('Water Volume (m³)')
plt.legend()
plt.show()

# Printing the predictions with district names
print("\nPredicted Volume of Water:")
print(df[['District', 'Total_Customer', 'Predicted_Volume_of_Water']])
