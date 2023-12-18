from skimage.feature import graycomatrix, graycoprops
import staintools
import numpy as np
from PIL import Image
import cv2
from skimage.filters import unsharp_mask
from skimage import color, filters, io, img_as_float
from skimage.transform import resize
from scipy.stats import skew, kurtosis

reference_image_path = './data/Adenocarcinoma/3dff0129-d23a-4496-bafc-1e8abe99439e.jpg'
reference_image = staintools.read_image(reference_image_path)
reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(reference_image)

def preprocess_image(image_path):
    """
    Applies unsharp masking to the input images
    """
    image = io.imread(image_path)
    #image = resize(image, (256,256))
    image = img_as_float(image)
    sharpened_image = unsharp_mask(image, radius=2, amount=5, preserve_range=True, channel_axis=2)
    return sharpened_image

def glcm_feature_extractor(grayscale_image, angles):
    """
    Implements the glcm to the input gray scale image and returns the dictionary of features
    """
    glcm = {
        'contrast': [],
        'dissimilarity': [],
        'homogeneity': [],
        'energy': [],
        'correlation': []
    }
    #conversion to match graycomatrix input type
    if grayscale_image.dtype == np.float32 or grayscale_image.dtype == np.float64:
        grayscale_image = (255 * grayscale_image).astype(np.uint8)  #glcm: input format: uint8 image

    for angle in angles:
        glcm_matrix = graycomatrix(grayscale_image, distances=[1], angles=[angle], symmetric=True, normed=True)
        # print(glcm)
        # print(graycoprops(glcm))
        # time.sleep(100)
        glcm['contrast'].append(graycoprops(glcm_matrix, 'contrast')[0, 0])
        glcm['dissimilarity'].append(graycoprops(glcm_matrix, 'dissimilarity')[0, 0])
        glcm['homogeneity'].append(graycoprops(glcm_matrix, 'homogeneity')[0, 0])
        glcm['energy'].append(graycoprops(glcm_matrix, 'energy')[0, 0])
        glcm['correlation'].append(graycoprops(glcm_matrix, 'correlation')[0, 0])

    return glcm

def hu_invariant_moments(grayscale_image):
    """
    Hue invariant moments to the input gray scale image and returns the dictionary of features
    """
    moments = cv2.moments(grayscale_image)
    hu_moments = cv2.HuMoments(moments)

    for i in range(0,7):
        hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
    return hu_moments.flatten()

def histogram_statistical_features_RGB(image, bins = 10):
    '''
    RGB
    '''
    color_hist = {}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red_hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    green_hist = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    blue_hist = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

    red_mean, red_median, red_std = np.mean(red_hist), np.median(red_hist), np.std(red_hist)
    green_mean, green_median, green_std = np.mean(green_hist), np.median(green_hist), np.std(green_hist)
    blue_mean, blue_median, blue_std = np.mean(blue_hist), np.median(blue_hist), np.std(blue_hist)

    color_hist['red_median'] = red_median
    color_hist['red_std'] = red_std
    color_hist['red_skewness'] = skew(red_hist)
    color_hist['red_kurtosis'] = kurtosis(red_hist)

    color_hist['green_median'] = green_median
    color_hist['green_std'] = green_std
    color_hist['green_skewness'] = skew(green_hist)
    color_hist['green_kurtosis'] = kurtosis(green_hist)

    color_hist['blue_median'] = blue_median
    color_hist['blue_std'] = blue_std
    color_hist['blue_skewness'] = skew(blue_hist)
    color_hist['blue_kurtosis'] = kurtosis(blue_hist)

    return color_hist

def histogram_statistical_features_HSV(image, bins = 10):
    '''HSV space'''

    color_hist = {}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue_hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    saturation_hist = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    value_hist = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

    hue_mean, hue_median, hue_std = np.mean(hue_hist), np.median(hue_hist), np.std(hue_hist)
    saturation_mean, saturation_median, saturation_std = np.mean(saturation_hist), np.median(saturation_hist), np.std(saturation_hist)
    value_mean, value_median, value_std = np.mean(value_hist), np.median(value_hist), np.std(value_hist)

    color_hist['hue_median'] = hue_median
    color_hist['hue_std'] = hue_std
    color_hist['hue_skewness'] = skew(hue_hist)
    color_hist['hue_kurtosis'] = kurtosis(hue_hist)

    color_hist['saturation_median'] = saturation_median
    color_hist['saturation_std'] = saturation_std
    color_hist['saturation_skewness'] = skew(saturation_hist)
    color_hist['saturation_kurtosis'] = kurtosis(saturation_hist)

    color_hist['value_median'] = value_median
    color_hist['value_std'] = value_std
    color_hist['value_skewness'] = skew(value_hist)
    color_hist['value_kurtosis'] = kurtosis(value_hist)

    return color_hist

def histogram_statistical_features_LAB(image, bins = 10):
    '''LAB space'''
    color_hist = {}
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    L_hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    A_hist = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    B_hist = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

    L_mean, L_median, L_std = np.mean(L_hist), np.median(L_hist), np.std(L_hist)
    A_mean, A_median, A_std = np.mean(A_hist), np.median(A_hist), np.std(A_hist)
    B_mean, B_median, B_std = np.mean(B_hist), np.median(B_hist), np.std(B_hist)

    color_hist['L_median'] = L_median
    color_hist['L_std'] = L_std
    color_hist['L_skewness'] = skew(L_hist)
    color_hist['L_kurtosis'] = kurtosis(L_hist)

    color_hist['A_median'] = A_median
    color_hist['A_std'] = A_std
    color_hist['A_skewness'] = skew(A_hist)
    color_hist['A_kurtosis'] = kurtosis(A_hist)

    color_hist['B_median'] = B_median
    color_hist['B_std'] = B_std
    color_hist['B_skewness'] = skew(B_hist)
    color_hist['B_kurtosis'] = kurtosis(B_hist)

    return color_hist

def extract_dft_features(preprocessed_image):
    '''
    Extracts the DFT features from the gray scale(preprocessed) image
    '''
    
    dft = np.fft.fft2(preprocessed_image)
    dft_shift = np.fft.fftshift(dft) 

    magnitude_spectrum = 20*np.log(np.abs(dft_shift)+1)

    mean, median, std_dev = np.mean(magnitude_spectrum), np.median(magnitude_spectrum), np.std(magnitude_spectrum)
    energy = np.sum(np.square(magnitude_spectrum)**2)
    
    return mean, median, std_dev, energy