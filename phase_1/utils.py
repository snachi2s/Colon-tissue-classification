from skimage.feature import graycomatrix, graycoprops
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy
import time


def preprocess_image(image_path):
    """
    Does resizing of the input image \n
    returns: resized image array
    """
    image = Image.open(image_path)
    image = image.resize((128,128))
    image = np.array(image)
    return image

def color_feature_extractor(image_array, bins=8):
    """
    Converts the input image array to HSV and returns the flattened histogram
    :param image_array: input image array (3 channeled)
    
    :returns histogram: flattened histogram 
    """
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(images=[hsv_image], channels=[0, 1, 2], mask=None, histSize=[bins, bins, bins], ranges=[0, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def glcm_feature_extractor(gray_image):
    """
    Computes GLCM and extracts textural features
    :param gray_image: input gray scale image
    :returns averaged_features: dictionary of averaged features
    """
    distances = [1]  
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  

    glcm = graycomatrix(gray_image, distances, angles, 256, symmetric=True, normed=True) #output:P[i,j,d,theta] --> [256,256,1,4]
    #normed set to true --> so that the proboabilities sum to 1
    features= {
        'Contrast': [],
        'Dissimilarity': [],
        'Homogeneity': [],
        'Energy': [],
        'Correlation': [],
        'ASM': [],
        'Entropy': [],
        'Variance': []
    }

    for i in range(len(angles)):
        features['Contrast'].append(graycoprops(glcm, 'contrast')[0, i])
        features['Dissimilarity'].append(graycoprops(glcm, 'dissimilarity')[0, i])
        features['Homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, i])
        features['Energy'].append(graycoprops(glcm, 'energy')[0, i])
        features['Correlation'].append(graycoprops(glcm, 'correlation')[0, i])
        features['ASM'].append(graycoprops(glcm, 'ASM')[0, i])
        #ref: https://github.com/luispedro/mahotas/blob/master/mahotas/features/texture.py
        glcm_ent = entropy(glcm[:, :, 0, i].ravel())
        features['Entropy'].append(glcm_ent)
        features['Variance'].append(np.var(glcm[:, :, 0, i]))
        
    averaged_features = {name: np.mean(values) for name, values in features.items()}

    return averaged_features

def hu_invariant_moments(grayscale_image):
    """
    Hue invariant moments to the input gray scale image and returns the dictionary of features
    """
    moments = cv2.moments(grayscale_image)
    hu_moments = cv2.HuMoments(moments)

    for i in range(0,7):
        hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
    return hu_moments.flatten()
