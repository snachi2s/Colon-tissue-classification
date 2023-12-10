import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm  # Import tqdm for progress bar
import os
import matplotlib.pyplot as plt

# Function to apply Canny edge detection to an image
def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Function for Histogram of Oriented Gradients (HOG) feature extraction
def extract_hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

# Load the CSV file into a DataFrame (provide path from your system)
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
        
        # Extract HOG features
        hog_features = extract_hog_features(edges)
        
        # Concatenate Canny edges and HOG features
        features = np.concatenate((edges.flatten(), hog_features))
        
        X.append(features)
        y.append(row['label'])

    return np.array(X), np.array(y)

# Specify the path to your images directory (provide path from your system)
images_directory = r'C:\Users\Manav\Desktop\code\ism_project_2023\train'

# Load and preprocess the images
X, y = load_images_and_labels(data_df, images_directory)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# # Display some example results
# for i in range(5):
#     sample_index = np.random.randint(len(X_test))
#     sample_image = X_test[sample_index][:256*256].reshape(256, 256)  # Extract Canny edges for visualization
#     predicted_label = classifier.predict([X_test[sample_index]])

#     plt.subplot(1, 5, i + 1)
#     plt.imshow(sample_image, cmap='gray')
#     plt.title(f"True: {y_test[sample_index]}\nPredicted: {predicted_label[0]}")
#     plt.axis('off')

# plt.show()
