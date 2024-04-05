import boto3
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error
from math import sqrt
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
    # Extract week number from date
    df['Week'] = df['Date'].dt.week
    # Drop unnecessary columns
    df.drop(columns=['Date'], inplace=True)
    return df

# Train recommender system model
def train_model(df):
    reader = Reader(rating_scale=(0, df['Quantity'].max()))
    data = Dataset.load_from_df(df[['Product_ID', 'Week', 'Quantity']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

# Generate recommendations for each week
def generate_recommendations(model, df):
    recommendations = {}
    for week in df['Week'].unique():
        week_data = df[df['Week'] == week]
        week_products = week_data['Product_ID'].unique()
        predictions = []
        for product_id in week_products:
            prediction = model.predict(product_id, week)
            predictions.append((product_id, prediction.est))
        top_products = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]  # Get top 5 products
        recommendations[week] = top_products
    return recommendations

# Evaluate model reliability and certainty
def evaluate_model_reliability(model, df):
    predictions = []
    for index, row in df.iterrows():
        prediction = model.predict(row['Product_ID'], row['Week'])
        predictions.append(prediction.est)
    actual = df['Quantity'].tolist()
    rmse = sqrt(mean_squared_error(actual, predictions))
    return rmse

# Evaluate errors and export to S3
def handle_errors(errors, folder_name):
    if errors:
        errors_df = pd.DataFrame(errors, columns=['Product_ID', 'Week', 'Error'])
        csv_buffer = io.StringIO()
        errors_df.to_csv(csv_buffer, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket_name, f"{folder_name}/errors.csv").put(Body=csv_buffer.getvalue())

# Validate data integrity and handle errors
def validate_data_integrity(df, folder_name):
    errors = []
    # Check for data format errors
    if not df['Product_ID'].dtype == 'int64':
        errors.append("Invalid data format: Product_ID")
    if not df['Week'].dtype == 'int64':
        errors.append("Invalid data format: Week")
    if not df['Quantity'].dtype == 'int64':
        errors.append("Invalid data format: Quantity")
    # Check for duplicates
    if df.duplicated().any():
        errors.append("Duplicate rows found")
    # Export errors to S3
    handle_errors(errors, folder_name)

# Export final result to S3
def export_to_s3(result, folder_name):
    result_df = pd.DataFrame(result)
    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, f"{folder_name}/recommendations.csv").put(Body=csv_buffer.getvalue())

# Main function
def main():
    # Load data from S3
    df = load_data_from_s3(bucket_name, folder_name)
    # Validate data integrity
    validate_data_integrity(df, 'Errors data load')
    # Preprocess data
    df = preprocess_data(df)
    # Train recommender system model
    model = train_model(df)
    # Generate recommendations
    recommendations = generate_recommendations(model, df)
    # Evaluate model reliability and certainty
    rmse = evaluate_model_reliability(model, df)
    print("Root Mean Squared Error (RMSE):", rmse)
    # Export recommendations to S3
    export_to_s3(recommendations, 'ML Production')

if __name__ == "__main__":
    main()
