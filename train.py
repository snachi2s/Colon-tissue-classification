import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.filters import unsharp_mask
import staintools
import time
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#from sklearn.model_selection import train_test_split
import joblib

def preprocess_image(image_path):
    """
    Applies unsharp masking to the input images
    """

    image = Image.open(image_path)
    image = np.array(image.resize((200,200)))

    unsharped_image = unsharp_mask(image, radius=2, amount=5, channel_axis=2)
    unsharped_image = (unsharped_image * 255).astype(np.uint8)
    return unsharped_image
    #normalized_image = normalizer.transform(unsharped_image)
    #return normalized_image

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

def feature_extraction(preprocessed_image):
    """
    Converts the pre-processed image to gray scale and calls feature extractor
    """
    gray_processed = cv2.cvtColor(np.float32(preprocessed_image), cv2.COLOR_BGR2GRAY)

    mean, median, std_dev = np.mean(gray_processed), np.median(gray_processed), np.std(gray_processed)
    
    glcm_features = glcm_feature_extractor(grayscale_image=gray_processed, angles=[0, np.pi/4, np.pi/2, 0.75*np.pi])
    #print(glcm_features)
    return mean, median, std_dev, glcm_features


def main():
    features = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Processed Images"):
        image_path = './train/' + row['name'] + '.jpg'
        #print(image_path)
        #time.sleep(8)
        preprocessed_image = preprocess_image(image_path)
        #print(type(preprocessed_image))
        #time.sleep(10)
        mean, median, std_dev, glcm_feature_dict = feature_extraction(preprocessed_image)
        one_part = ({
            'image_id': row['name'],
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'label': row['label'],
        })

        for feature, value in glcm_feature_dict.items():
            for index, value in enumerate(value):
                column_name = f"{feature}{index+1}"
                one_part[column_name] = value
        features.append(one_part)

    features_dataframe = pd.DataFrame(features)
    #print(features_dataframe.head())

    #######################TRAINING###################
    
    X = features_dataframe.drop(['image_id', 'label'], axis=1)
    print(len(X))
    y = features_dataframe['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    classifier = SVC(kernel='rbf', verbose=True) 
    classifier.fit(X_train_scaled, y)

    joblib.dump(classifier, 'svm_trained.pkl')
    
    #metrics
    train_predictions = classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_true=y, y_pred=train_predictions)
    print("Training accuracy: ", train_accuracy)

    conf_matrix = confusion_matrix(y, train_predictions)
    print(conf_matrix)

    report = classification_report(y, train_predictions)
    print(report)

if __name__ == '__main__':
    CSV_PATH = 'train.csv'
    IMG_DIR = './train'
    
    dataframe = pd.read_csv(CSV_PATH)

    #not used (in accordance with paper)
    reference_image_path = './data/Adenocarcinoma/3dff0129-d23a-4496-bafc-1e8abe99439e.jpg'
    reference_image = staintools.read_image(reference_image_path)
    reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(reference_image)
    
    main()


##subplots
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(Image.open(image_path))
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# axes[1].imshow(preprocessed_image, cmap='gray')
# axes[1].set_title('Preprocessed Image')
# axes[1].axis('off')

# plt.show()
