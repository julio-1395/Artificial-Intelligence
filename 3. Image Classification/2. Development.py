import boto3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', aws_secret_access_key='YOUR_SECRET_KEY')

# Specify the bucket name and folder name in S3
bucket_name = 'your-bucket-name'
folder_name = 'animals/'

# Download images from S3 bucket
def download_images_from_s3(bucket_name, folder_name):
    images = []
    labels = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('.jpg') or key.endswith('.jpeg') or key.endswith('.png'):
            obj = s3.get_object(Bucket=bucket_name, Key=key)
            image = Image.open(io.BytesIO(obj['Body'].read()))
            image = image.resize((224, 224))  # Resize image to a fixed size
            images.append(np.array(image))
            label = key.split('/')[1]  # Extract label from file path
            labels.append(label)
    return np.array(images), np.array(labels)

# Preprocess images
def preprocess_images(images):
    # Normalize pixel values to the range [0, 1]
    images = images / 255.0
    return images

# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Download images and labels
images, labels = download_images_from_s3(bucket_name, folder_name)

# Preprocess images
images = preprocess_images(images)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build and compile the model
model = build_model(input_shape=(224, 224, 3), num_classes=5)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the trained model
model.save("animal_classification_model.h5")
