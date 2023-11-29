import cv2
import numpy as np
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set the path for saving images
output_directory = r'C:\Users\DELL\Desktop\ISM\ism_project_2023\filtered_images'
create_directory(output_directory)

# Load the image
img = cv2.imread(r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Common parameters
phi = 0

# Iterate through ksize values
for ksize in range(10, 101):
    # Iterate through sigma values
    for sigma in np.arange(0.1, 5.1, 0.1):
        # Iterate through theta values
        for theta in np.arange(0, np.pi + 0.1, np.pi/20):
            # Iterate through lamda values
            for lamda in range(1, 11):
                # Iterate through gamma values
                for gamma in np.arange(0.1, 3.1, 0.1):
                    # Generate Gabor kernel
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

                    # Apply Gabor filter to the image
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

                    # Display and save images
                    image_id = f"ksize_{ksize}_sigma_{sigma:.2f}_theta_{theta:.2f}_lamda_{lamda}_gamma_{gamma:.2f}"
                    # cv2.imshow('filtered image', fimg)
                    cv2.imwrite(os.path.join(output_directory, f"{image_id}.jpg"), fimg)

                    # Print parameters with image id
                    print(f"{image_id} - ksize: {ksize}, sigma: {sigma:.2f}, theta: {theta:.2f}, lamda: {lamda}, gamma: {gamma:.2f}")

                    # Uncomment the following line if you want to wait for a key press before displaying the next image
                    # cv2.waitKey(0)

# Uncomment the following line if you want to close the OpenCV windows at the end
# cv2.destroyAllWindows()
