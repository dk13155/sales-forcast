import pandas as pd
from prophet import Prophet
import pickle
import os

# --- Load CSV ---
file_path = 'C:/Users/User/OneDrive/Desktop/Intership/sales_forecast_app/data/retail_sales_dataset.csv'  # Make sure your CSV is here
data = pd.read_csv(file_path, parse_dates=['Date'])

# --- Aggregate daily sales ---
sales_data = data.groupby('Date')['Total Amount'].sum().reset_index()
sales_data.rename(columns={'Date':'ds', 'Total Amount':'y'}, inplace=True)

# --- Train Prophet model ---
model = Prophet(daily_seasonality=True)
model.fit(sales_data)

# --- Create models folder ---
if not os.path.exists('models'):
    os.makedirs('models')

# --- Save trained model ---
model_file = 'models/prophet_sales_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Prophet model trained and saved as '{model_file}'")
