import boto3
import pandas as pd
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import io

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', aws_secret_access_key='YOUR_SECRET_KEY')

# Specify the bucket name and folder name in S3
bucket_name = 'your-bucket-name'
folder_name = 'Dataset/'

# Load and concatenate CSV files from S3
def load_data_from_s3(bucket_name, folder_name):
    dfs = []
    for month in range(1, 13):
        key = folder_name + f"{month}.csv"
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Preprocess data
def preprocess_data(df):
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Extract quarter from date
    df['Quarter'] = df['Date'].dt.quarter
    # Drop unnecessary columns
    df.drop(columns=['Date'], inplace=True)
    return df

# Forecast transport costs by quarter using ARIMA
def forecast_transport_costs(df):
    quarters = sorted(df['Quarter'].unique())
    forecasts = {}
    for quarter in quarters:
        quarter_data = df[df['Quarter'] == quarter]
        transport_costs = quarter_data['Transport_Costs'].tolist()
        model = ARIMA(transport_costs, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=4)  # Forecast for the next 4 quarters
        forecasts[quarter] = forecast
    return forecasts

# Evaluate model reliability and certainty
def evaluate_model_reliability(actual, forecasts):
    all_actual = []
    all_forecasts = []
    for quarter in actual.keys():
        all_actual.extend(actual[quarter])
        all_forecasts.extend(forecasts[quarter])
    rmse = np.sqrt(mean_squared_error(all_actual, all_forecasts))
    return rmse

# Export final result to S3
def export_to_s3(result):
    result_df = pd.DataFrame(result)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, 'ML Production/forecasted_transport_costs.csv').put(Body=csv_buffer.getvalue())

# Main function
def main():
    # Load data from S3
    df = load_data_from_s3(bucket_name, folder_name)
    # Preprocess data
    df = preprocess_data(df)
    # Forecast transport costs by quarter
    forecasts = forecast_transport_costs(df)
    # Evaluate model reliability and certainty
    rmse = evaluate_model_reliability(actual, forecasts)
    print("Root Mean Squared Error (RMSE):", rmse)
    # Export final result to S3
    export_to_s3(forecasts)

if __name__ == "__main__":
    main()
