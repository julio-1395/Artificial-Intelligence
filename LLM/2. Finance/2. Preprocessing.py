# Function to download and preprocess text data from S3
def preprocess_data_from_s3(bucket_name, folder_name):
    # List objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    # Initialize an empty list to store preprocessed data
    preprocessed_data = []

    # Iterate through objects in the folder
    for obj in response['Contents']:
        # Download the object
        file_key = obj['Key']
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = obj['Body'].read().decode('utf-8')
        
        # Preprocess text data (customize this based on your requirements)
        cleaned_text = preprocess_text(content)
        
        # Append preprocessed text to the list
        preprocessed_data.append(cleaned_text)

    return preprocessed_data

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    return text

# Preprocess text data from S3
preprocessed_data = preprocess_data_from_s3(bucket_name, folder_name)

# Example printing the first item in the preprocessed data list
print(preprocessed_data[0])