# trainmodel.py

import pandas as pd
import pickle
import os
from prophet import Prophet


def train_model(csv_path):
    """
    Train Prophet model using CSV file
    CSV must contain:
    - Date
    - Total Amount
    """

    # Load data
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Aggregate daily sales
    df = df.groupby('Date')['Total Amount'].sum().reset_index()

    # Rename columns for Prophet
    df.rename(columns={'Date': 'ds', 'Total Amount': 'y'}, inplace=True)

    # Initialize Prophet model
    model = Prophet(daily_seasonality=True)

    # Train model
    model.fit(df)

    

    # Save trained model
    with open("prophet_sales_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved successfully!")


# Optional runner
if __name__ == "__main__":
    train_model("C:/Users/User/OneDrive/Desktop/Intership/sales_forecast_app/data/retail_sales_dataset.csv")  # Change filename if needed
