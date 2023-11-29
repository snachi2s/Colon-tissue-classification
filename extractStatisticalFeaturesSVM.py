import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.filters import unsharp_mask

# Path to CSV file
csv_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\train.csv'

# Path to image folder
image_folder = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\train'

# Read CSV file
df = pd.read_csv(csv_path)

# Lists to store features and labels
features = []
labels = []

# Iterate through each row in the CSV file
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
    image_name = row['name'] + ".jpg"
    label = row['label']
    
    # Construct the full path to the image
    image_path = os.path.join(image_folder, image_name)
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    
    # Apply unsharp mask
    unsharped_image = unsharp_mask(image, radius=2, amount=5, channel_axis=2)  
    
    # Convert to grayscale
    gray_processed = cv2.cvtColor(np.float32(unsharped_image), cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    resized_image = cv2.resize(gray_processed, (200, 200))
    
    # Normalize the image
    normalized_image = resized_image / 255.0
    
    # Extract statistical features
    features_vector = [                     # each feature is a column
        skew(normalized_image.flatten()),
        kurtosis(normalized_image.flatten()),
        np.percentile(normalized_image, 25),
        np.percentile(normalized_image, 50),
        np.percentile(normalized_image, 75),    
    ]
    
    # Append the features and label to the lists
    features.append(features_vector)
    labels.append(label)

# Convert lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_std, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test_std)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
