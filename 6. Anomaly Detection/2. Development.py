import boto3
import pandas as pd
from sklearn.ensemble import IsolationForest
import io
from datetime import datetime, timedelta
import time

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
    # Drop unnecessary columns
    df.drop(columns=['Date', 'Time', 'Currency', 'Merchant_Name', 'Reference_Number',
                     'Card_Type', 'Card_Number', 'Authorization_Code', 'Customer_Name',
                     'Branch_Code', 'Transaction_Status'], inplace=True)
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Transaction_Type', 'Merchant_Category', 'Location', 'Channel'])
    return df

# Train anomaly detection model
def train_anomaly_detection_model(df):
    model = IsolationForest(contamination=0.01)  # 1% of data considered as outliers
    model.fit(df)
    return model

# Detect anomalies in data
def detect_anomalies(model, df):
    predictions = model.predict(df)
    anomalies = df[predictions == -1]  # Select rows predicted as anomalies
    return anomalies

# Evaluate model reliability and certainty
def evaluate_model_reliability(anomalies):
    # Calculate proportion of anomalies
    anomaly_rate = len(anomalies) / len(df)
    return anomaly_rate

# Export final result to S3
def export_to_s3(result):
    result_df = pd.DataFrame(result)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, 'ML Production/anomalies.csv').put(Body=csv_buffer.getvalue())

# Schedule deployment of the model
def schedule_model_deployment():
    # Calculate the time 3 hours from now
    deploy_time = datetime.now() + timedelta(hours=3)
    # Schedule deployment on weekdays (Monday to Friday)
    while deploy_time.weekday() >= 5:  # 5 represents Saturday, 6 represents Sunday
        deploy_time += timedelta(days=1)  # Move to the next weekday
    # Print scheduled deployment time
    print("Scheduled deployment time:", deploy_time)
    # Wait until deployment time
    time.sleep((deploy_time - datetime.now()).total_seconds())
    # Call function to deploy the model
    deploy_model()

# Deploy the model
def deploy_model():
    # Load data from S3
    df = load_data_from_s3(bucket_name, folder_name)
    # Preprocess data
    df = preprocess_data(df)
    # Train anomaly detection model
    model = train_anomaly_detection_model(df)
    # Detect anomalies
    anomalies = detect_anomalies(model, df)
    # Evaluate model reliability and certainty
    anomaly_rate = evaluate_model_reliability(anomalies)
    print("Anomaly Rate:", anomaly_rate)
    # Export final result to S3
    export_to_s3(anomalies)

# Main function
def main():
    # Schedule deployment of the model
    schedule_model_deployment()

if __name__ == "__main__":
    main()
