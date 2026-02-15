# train_model.py

import pandas as pd
import pickle
from prophet import Prophet


def train_model(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Aggregate sales per day
    df = df.groupby("Date")["Total Amount"].sum().reset_index()

    # Rename columns for Prophet
    df.rename(columns={"Date": "ds", "Total Amount": "y"}, inplace=True)

    # Create model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Save model in ROOT folder (important)
    with open("prophet_sales_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved as prophet_sales_model.pkl")


if __name__ == "__main__":
    train_model("C:/Users/User/OneDrive/Desktop/Intership/sales_forecast_app/data/retail_sales_dataset.csv")
