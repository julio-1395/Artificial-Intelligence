import pandas as pd
import numpy as np
import boto3
from io import StringIO

def load_data_from_s3(bucket_name, folder_name):
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # List all objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    
    # Initialize an empty DataFrame to store data
    df_list = []
    
    # Loop through each object in the folder
    for obj in response['Contents']:
        # Extract the file name
        file_name = obj['Key'].split('/')[-1]
        
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            # Load the CSV file from S3 into a DataFrame
            obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            df = pd.read_csv(obj['Body'])
            
            # Append the DataFrame to the list
            df_list.append(df)
    
    # Concatenate all DataFrames into one
    merged_df = pd.concat(df_list, ignore_index=True)
    
    return merged_df

def preprocess_data(df):
    # Drop duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    return df

def recommend_top_rated_books_by_genre(df):
    # Group by genre and calculate average rating for each product
    genre_avg_rating = df.groupby(['Genre', 'Product_ID'])['Rating'].mean().reset_index()
    
    # Sort by rating in descending order for each genre
    genre_avg_rating.sort_values(by=['Genre', 'Rating'], ascending=[True, False], inplace=True)
    
    # Get top 5 rated books for each genre
    top_rated_books_by_genre = genre_avg_rating.groupby('Genre').head(5)
    
    return top_rated_books_by_genre

def evaluate_model():
    # Placeholder for model evaluation code
    pass

def export_errors_to_s3(errors_df, bucket_name, folder_name):
    # Initialize S3 client
    s3 = boto3.client('s3')
    
    # Convert DataFrame to CSV format
    csv_buffer = StringIO()
    errors_df.to_csv(csv_buffer, index=False)
    
    # Upload CSV file to S3
    response = s3.put_object(
        Bucket=bucket_name,
        Key=f"{folder_name}/errors.csv",
        Body=csv_buffer.getvalue()
    )

def main():
    # Load data from S3
    bucket_name = 'your_bucket_name'
    folder_name = 'Sales_2023'
    sales_data = load_data_from_s3(bucket_name, folder_name)
    
    # Preprocess data
    sales_data = preprocess_data(sales_data)
    
    # Recommend top rated books by genre
    top_rated_books = recommend_top_rated_books_by_genre(sales_data)
    
    # Evaluate model
    evaluate_model()
    
    # Export errors (if any) to S3
    errors_df = pd.DataFrame()  # Placeholder for errors DataFrame
    export_errors_to_s3(errors_df, bucket_name, 'Errors_Recommendation_Model')
    
    # Export final result to S3
    top_rated_books.to_csv('top_rated_books.csv', index=False)
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file('top_rated_books.csv', f'ML_Production/top_rated_books.csv')
    
if __name__ == "__main__":
    main()
