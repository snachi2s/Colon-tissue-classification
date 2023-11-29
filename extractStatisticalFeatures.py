from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis

# Path to the image
image_path = r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg'

# Open the image using PIL
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Statistical feature extraction
mean_intensity = np.mean(image_array)
median_intensity = np.median(image_array)
std_dev_intensity = np.std(image_array)
variance_intensity = np.var(image_array)
skewness_intensity = skew(image_array.flatten())
kurtosis_intensity = kurtosis(image_array.flatten())
min_intensity = np.min(image_array)
max_intensity = np.max(image_array)
range_intensity = np.ptp(image_array)
percentile_25 = np.percentile(image_array, 25)
percentile_50 = np.percentile(image_array, 50)
percentile_75 = np.percentile(image_array, 75)

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
