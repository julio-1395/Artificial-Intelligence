import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', aws_secret_access_key='YOUR_SECRET_KEY')

# Specify the bucket name and folder name in S3
bucket_name = 'your-bucket-name'
folder_name = 'Dataset/'

# Function to download data from S3
def download_data_from_s3(bucket_name, folder_name):
    # List objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    # Initialize an empty list to store data
    data = []

    # Iterate through objects in the folder
    for obj in response['Contents']:
        # Download the object
        file_key = obj['Key']
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = obj['Body'].read().decode('utf-8')
        
        # Append content to the data list
        data.append(content)

    return data

# Preprocess text data
def preprocess_text(text):
    # Example preprocessing steps: tokenization, removing punctuation, lowercasing, etc.
    return text.lower()

# Download data from S3
data = download_data_from_s3(bucket_name, folder_name)

# Preprocess the data
preprocessed_data = [preprocess_text(text) for text in data]

# Assuming your data includes both text and labels
# Split the data into input text and labels
texts = preprocessed_data['text']
labels = preprocessed_data['label']

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Encode labels (assuming labels are categorical)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=3)])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Optionally, deploy the model for inference
# Example: Save the trained model
model.save("insurance_nlp_model.h5")
