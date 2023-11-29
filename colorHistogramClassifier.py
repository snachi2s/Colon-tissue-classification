from tqdm import tqdm
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to compute color features for an image
def compute_color_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at path {image_path}")
        return None, None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    return hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()

# Load CSV file containing image names and labels
csv_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\train.csv'
df = pd.read_csv(csv_path)

# Path to the folder containing training images
image_folder_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\train'

# Initialize lists to store features and labels
features = []
labels = []

# Use an alias for tqdm
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
    image_name = row['name'] + ".jpg"  # Add the file extension
    label = row['label']

    # Construct the full path to the image using raw string
    image_path = os.path.join(image_folder_path, image_name)

    # Compute color features for the current image
    hist_red, hist_green, hist_blue = compute_color_features(image_path)
    
    if hist_red is not None:
        # Concatenate features into a single vector
        feature_vector = hist_red.tolist() + hist_green.tolist() + hist_blue.tolist()

        features.append(feature_vector)
        labels.append(label)


# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
