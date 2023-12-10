import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pywt     # pip install PyWavelets
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Function to apply Canny edge detection to an image
def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Function for Haar wavelet feature extraction
def extract_haar_wavelet_features(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = np.concatenate((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))
    return features

# Load the CSV file into a DataFrame
csv_path = r'C:\Users\Manav\Desktop\code\ism_project_2023\train.csv'
data_df = pd.read_csv(csv_path)

# Load and preprocess the images
def load_images_and_labels(data_df, images_dir):
    X = []
    y = []

    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing images"):
        image_path = os.path.join(images_dir, row['name'])
        image = cv2.imread(image_path + '.jpg')
        
        # Apply Canny edge detection
        edges = apply_canny(image)
        print(f"Step 1: Canny edge detection applied on {row['name']}")

        # Extract Haar wavelet features
        wavelet_features = extract_haar_wavelet_features(edges)
        print(f"Step 2: Haar wavelet features extracted from {row['name']}")

        # Concatenate Canny edges and Haar wavelet features
        features = np.concatenate((edges.flatten(), wavelet_features))
        print(f"Step 3: Feature vector created for {row['name']}")

        X.append(features)
        y.append(row['label'])

    return np.array(X), np.array(y)

# Specify the path to your images directory
images_directory = r'C:\Users\Manav\Desktop\code\ism_project_2023\train'

# Load and preprocess the images
X, y = load_images_and_labels(data_df, images_directory)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
print("Training the classifier...")
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict on the test set
print("Predicting on the test set...")
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display some example results
for i in range(5):
    sample_index = np.random.randint(len(X_test))
    sample_image = X_test[sample_index][:256*256].reshape(256, 256)  # Extract Canny edges for visualization
    predicted_label = classifier.predict([X_test[sample_index]])

    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"True: {y_test[sample_index]}\nPredicted: {predicted_label[0]}")
    plt.axis('off')

plt.show()
