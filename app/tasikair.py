import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Disable scientific notation in pandas
pd.set_option('display.float_format', '{:.2f}'.format)

# Importing data from the CSV file
df = pd.read_csv('data_air_bersih.csv')

# Renaming columns
df.columns = ['District', 'Volume_Water_m3', 'Total_Customer', 'Total_Resident']

# Displaying the imported data
print("Imported Data with Updated Columns:")
print(df)

# Displaying the available columns in the DataFrame
print("\nAvailable Columns in DataFrame:")
print(df.columns)

# Defining independent (X) and dependent (y) variables
X = df[['Total_Customer', 'Total_Resident']]  # Independent variables
y = df['Volume_Water_m3']  # Dependent variable

# Adding a constant to the model
X = sm.add_constant(X)

# Building the regression model
model = sm.OLS(y, X).fit()

# Viewing the model summary
print("\nRegression Model Summary:")
print(model.summary())

# Calculating model accuracy
mse = mean_squared_error(y, model.predict(X))
r_squared = r2_score(y, model.predict(X))

# Displaying model accuracy
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r_squared:.2f}")

# Creating a scenario where the number of customers equals the number of residents
df_scenario = df.copy()  # Copying the original DataFrame
df_scenario['Total_Customer'] = df_scenario['Total_Resident']  # Setting customers = residents

# Preparing the X variable for prediction based on the scenario
X_scenario = df_scenario[['Total_Customer', 'Total_Resident']]
X_scenario = sm.add_constant(X_scenario)  # Adding a constant

# Predicting based on the scenario
predictions_scenario = model.predict(X_scenario)

# Adding the prediction results to the DataFrame
df_scenario['Predicted_Volume_Water'] = predictions_scenario

# Displaying the prediction results for the scenario
print("\nPredicted Water Volume if All Residents Become Customers:")
print(df_scenario[['District', 'Total_Customer', 'Total_Resident', 'Predicted_Volume_Water']])

# Calculating the number of residents who are not yet customers
df['Non_Customer_Residents'] = df['Total_Resident'] - df['Total_Customer']

# Displaying data with the number of non-customer residents
print("\nData with Non-Customer Residents:")
print(df[['District', 'Total_Customer', 'Total_Resident', 'Non_Customer_Residents']])

# Visualizing prediction results and non-customer residents

# 1. Visualizing predicted water volume if all residents become customers
plt.figure(figsize=(10, 6))

# Scatter plot for predicted water volume based on the number of customers
plt.scatter(df_scenario['Total_Customer'], df_scenario['Predicted_Volume_Water'], color='green', label='Scenario Prediction', linewidth=2)

# Setting proper x and y axis values
plt.gca().ticklabel_format(style='plain', axis='x')  # Disable scientific notation on X axis
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Format Y axis as regular numbers (no decimals)

plt.title('Predicted Water Volume If All Residents Become Customers')
plt.xlabel('Number of Customers (Equal to Number of Residents)')
plt.ylabel('Water Volume (m³)')
plt.legend()
plt.show()

# 2. Visualizing comparison of non-customer residents and existing customers
plt.figure(figsize=(10, 6))
plt.bar(df['District'], df['Total_Customer'], color='blue', label='Customers', alpha=0.7)
plt.bar(df['District'], df['Non_Customer_Residents'], bottom=df['Total_Customer'], color='red', label='Non-Customer Residents', alpha=0.7)
plt.title('Comparison of Number of Customers and Non-Customer Residents by District')
plt.xlabel('District')
plt.ylabel('Number of People')
plt.xticks(rotation=45, ha='right')  # Rotating the district labels for better readability
plt.legend()
plt.show()
