import os
import cv2
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Load the trained SVM model
model_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\color_histogram_svm_model.joblib'
svm_clf = joblib.load(model_path)

# Path to the folder containing test images
test_image_folder_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\test'

# Load CSV file containing test image names
test_csv_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\test.csv'
test_df = pd.read_csv(test_csv_path)

# Initialize lists to store features and image names
test_features = []
test_image_names = []

# Use an alias for tqdm
for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Images"):
    image_name = row['name']   # Add the file extension
    test_image_names.append(image_name)

    # Construct the full path to the test image using raw string
    test_image_path = os.path.join(test_image_folder_path, image_name + ".jpg")

    # Compute color features for the current test image
    hist_red, hist_green, hist_blue = compute_color_features(test_image_path)
    
    if hist_red is not None:
        # Concatenate features into a single vector
        feature_vector = hist_red.tolist() + hist_green.tolist() + hist_blue.tolist()

        test_features.append(feature_vector)

# Normalize test features
scaler = StandardScaler()
test_features_normalized = scaler.fit_transform(test_features)

# Make predictions on the test set
test_predictions = svm_clf.predict(test_features_normalized)

# Create a DataFrame to store results
result_df = pd.DataFrame({
    'name': test_image_names,
    'label': test_predictions
})

# Save the results to a CSV file
result_csv_path = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\test_predictions.csv'
result_df.to_csv(result_csv_path, index=False)

print(f"Predictions saved to: {result_csv_path}")
