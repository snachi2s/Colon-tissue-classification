from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis
import cv2

# Path to the image
image_path = r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg'

# Load the image
image = cv2.imread(image_path)

# Resize the image to a specific size 
resized_image = cv2.resize(image, (200, 200))

# # Convert the image to grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize pixel values to the range [0, 1]
normalized_image = resized_image / 255.0

# Statistical feature extraction
mean_intensity = np.mean(normalized_image)
median_intensity = np.median(normalized_image)
std_dev_intensity = np.std(normalized_image)
variance_intensity = np.var(normalized_image)
skewness_intensity = skew(normalized_image.flatten())
kurtosis_intensity = kurtosis(normalized_image.flatten())
min_intensity = np.min(normalized_image)
max_intensity = np.max(normalized_image)
range_intensity = np.ptp(normalized_image)
percentile_25 = np.percentile(normalized_image, 25)
percentile_50 = np.percentile(normalized_image, 50)
percentile_75 = np.percentile(normalized_image, 75)

# Print or use the extracted features as needed
print("Mean Intensity:", mean_intensity)
print("Median Intensity:", median_intensity)
print("Standard Deviation:", std_dev_intensity)
print("Variance:", variance_intensity)
print("Skewness:", skewness_intensity)
print("Kurtosis:", kurtosis_intensity)
print("Min Intensity:", min_intensity)
print("Max Intensity:", max_intensity)
print("Range:", range_intensity)
print("25th Percentile:", percentile_25)
print("50th Percentile (Median):", percentile_50)
print("75th Percentile:", percentile_75)
