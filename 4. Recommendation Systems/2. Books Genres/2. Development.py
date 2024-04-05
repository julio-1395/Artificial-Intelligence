import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import io

# Function to load data from S3
def load_data_from_s3(bucket_name, folder_name):
    s3 = boto3.client('s3')
    files = ['January.csv', 'February.csv', 'March.csv', 'April.csv', 'May.csv', 'June.csv', 'July.csv', 
             'August.csv', 'September.csv', 'October.csv', 'November.csv', 'December.csv']
    data_frames = []
    for file in files:
        obj = s3.get_object(Bucket=bucket_name, Key=f"{folder_name}/{file}")
        data_frames.append(pd.read_csv(obj['Body']))
    return pd.concat(data_frames, ignore_index=True)

# Function to preprocess data
def preprocess_data(data):
    # Drop irrelevant columns
    data = data[['Product_ID', 'Genre', 'Rating']]
    # Remove duplicates if any
    data.drop_duplicates(inplace=True)
    return data

# Function to train content-based recommendation model
def train_content_based_model(data):
    # Preprocess data
    data = preprocess_data(data)
    # Group by Genre and calculate mean rating
    genre_ratings = data.groupby('Genre')['Rating'].mean().reset_index()
    # Sort by rating in descending order
    genre_ratings = genre_ratings.sort_values(by='Rating', ascending=False)
    return genre_ratings

# Function to evaluate model reliability and certainty
def evaluate_model(genre_ratings):
    # Calculate mean rating across all genres
    mean_rating = genre_ratings['Rating'].mean()
    # Calculate standard deviation of ratings
    std_rating = genre_ratings['Rating'].std()
    return mean_rating, std_rating

# Function to export data to S3
def export_to_s3(data, bucket_name, folder_name, file_name):
    s3 = boto3.client('s3')
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=f"{folder_name}/{file_name}", Body=csv_buffer.getvalue())

# Load data from S3
bucket_name = 'your_bucket_name'
folder_name = 'Sales_2023'
data = load_data_from_s3(bucket_name, folder_name)

# Train content-based recommendation model
genre_ratings = train_content_based_model(data)

# Evaluate model reliability and certainty
mean_rating, std_rating = evaluate_model(genre_ratings)

# Export the final model to S3
export_to_s3(genre_ratings, bucket_name, 'ML Production', 'content_based_model.csv')
