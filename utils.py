from skimage.feature import graycomatrix, graycoprops
import staintools
import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask
import cv2

reference_image_path = './data/Adenocarcinoma/3dff0129-d23a-4496-bafc-1e8abe99439e.jpg'
reference_image = staintools.read_image(reference_image_path)
reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(reference_image)

def preprocess_image(image_path):
    """
    Applies unsharp masking to the input images
    """

    image = Image.open(image_path)
    image = np.array(image.resize((200,200)))

    unsharped_image = unsharp_mask(image, radius=2, amount=5, channel_axis=2)
    unsharped_image = (unsharped_image * 255).astype(np.uint8)
    return unsharped_image

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