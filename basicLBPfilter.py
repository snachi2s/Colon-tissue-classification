from skimage import feature, color
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Read the image and convert it to grayscale
image_path = r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Padding (optional)
# padded_image = ...

# Step 3: Divide the image into cells
cell_size = 4

# Step 4 and 5: Calculate LBP for each pixel and histogram for each cell
lbp_features = []
P = 8  # Number of neighbors
R = 2  # Radius
for i in range(0, gray_image.shape[0], cell_size):
    for j in range(0, gray_image.shape[1], cell_size):
        cell = gray_image[i:i+cell_size, j:j+cell_size]
        lbp = feature.local_binary_pattern(cell, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        lbp_features.extend(hist)

# Step 6: Concatenate histograms
feature_vector = np.array(lbp_features)

# Step 7: Normalization (optional)
feature_vector = feature_vector / np.linalg.norm(feature_vector)

# Print or use the feature vector as needed
print("LBP Feature Vector:", feature_vector)

# Display the original image and the LBP image for visualization
lbp_image = feature.local_binary_pattern(gray_image, P, R, method='uniform')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(lbp_image, cmap='gray')
axes[1].set_title('LBP Image')
plt.show()
