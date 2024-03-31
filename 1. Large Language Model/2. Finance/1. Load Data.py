import boto3
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', aws_secret_access_key='YOUR_SECRET_KEY')

# Specify the bucket name and folder name in S3
bucket_name = 'your-bucket-name'
folder_name = 'Data/'
