from skimage.feature import graycomatrix, graycoprops
import staintools
import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask
import cv2
from skimage import color, filters, io, img_as_float
from skimage.transform import resize

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
    image = resize(image, (256,256))
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

def extract_color_histogram_features(image_path, num_bins=128):
    '''
    Extracts histogram values only from the red channels of the input image
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red_hist = cv2.calcHist([image], [0], None, [num_bins], [0, 256]).flatten()

    red_hist = red_hist / red_hist.sum()
    return red_hist