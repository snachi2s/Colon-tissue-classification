import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import staintools
import time
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import lightgbm
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import utils

def feature_extraction(preprocessed_image):
    """
    Converts the pre-processed image to gray scale and calls feature extractor
    """
    gray_processed = cv2.cvtColor(np.float32(preprocessed_image), cv2.COLOR_BGR2GRAY)

    mean, median, std_dev = np.mean(gray_processed), np.median(gray_processed), np.std(gray_processed)
    percentile_25, percentile_75 = np.percentile(gray_processed, 25), np.percentile(gray_processed, 75)

    glcm_features = utils.glcm_feature_extractor(grayscale_image=gray_processed, angles=[0, np.pi/4, np.pi/2, 0.75*np.pi])
    #print(glcm_features)

    hu_moments = utils.hu_invariant_moments(grayscale_image=gray_processed)

    return mean, median, std_dev, percentile_25, percentile_75, glcm_features, hu_moments


def main():
    features = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Processed Images"):
        image_path = './train/' + row['name'] + '.jpg'
        #print(image_path)
        #time.sleep(8)
        preprocessed_image = utils.preprocess_image(image_path)
        #print(type(preprocessed_image))
        #time.sleep(10)
        mean, median, std_dev, per_25, per_75, glcm_feature_dict, hu_list = feature_extraction(preprocessed_image)
        one_part = ({
            'image_id': row['name'],
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'percentile_25': per_25,
            'percentile_75': per_75,
            'hu1': hu_list[0],
            'hu2': hu_list[1],
            'hu3': hu_list[2],
            'hu4': hu_list[3],
            'hu5': hu_list[4],
            'hu6': hu_list[5],
            'hu7': hu_list[6],
            'label': row['label'],
        })

        for feature, value in glcm_feature_dict.items():
            for index, value in enumerate(value):
                column_name = f"{feature}{index+1}"
                one_part[column_name] = value
        features.append(one_part)

    features_dataframe = pd.DataFrame(features)
    #print(features_dataframe.head())

    #separate the training images we have into train(80%) and test(20%) 
    X_train, X_test, y_train, y_test = train_test_split(features_dataframe.drop(['image_id', 'label'], axis=1), features_dataframe['label'], test_size=0.2, random_state=42)

    ####################TRAINING###################

    ####uncomment when want to find the best hyperparameters of one model 
    # param_grid = {
    #     'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': ['scale', 'auto'],
    #     'kernel': ['rbf', 'poly', 'sigmoid']
    # }

    #grid_search = GridSearchCV(SVC(), param_grid, verbose=3, refit=True)
    #grid_search.fit(X_train, y_train)

    #print("best hyperparameters:", grid_search.best_params_)
    #print("corresponding score:", grid_search.best_score_)
    #After finding the best hyperparameters, save/rememer the values and use them in the classifier

    ###Normal training 

    X_train_scaled = StandardScaler().fit_transform(X_train)
    y = y_train

    X_test_scaled = StandardScaler().fit_transform(X_test)
    y_test = y_test

    #TODO: Voting classifier --> SVM, XGBoost, LightGBM, Random Forest

    #svm
    classifier = SVC(kernel='rbf', verbose=True, gamma='scale', C=10)
    classifier.fit(X_train_scaled, y)

    #train metrics
    train_predictions = classifier.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_true=y, y_pred=train_predictions)
    print("Training accuracy: ", train_accuracy)

    #test
    test_predictions = classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_true=y_test, y_pred=test_predictions)
    print("Test accuracy: ", test_accuracy)

    #confusion matrix
    print("Confusion matrix: ")
    print(confusion_matrix(y_true=y_test, y_pred=test_predictions))

    #classification report
    print("Classification report: ")
    print(classification_report(y_true=y_test, y_pred=test_predictions))
    

if __name__ == '__main__':
    CSV_PATH = 'train.csv'
    IMG_DIR = './train'
    
    dataframe = pd.read_csv(CSV_PATH)

    #not used (in accordance with paper)
    # reference_image_path = './data/Adenocarcinoma/3dff0129-d23a-4496-bafc-1e8abe99439e.jpg'
    # reference_image = staintools.read_image(reference_image_path)
    # reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
    # normalizer = staintools.StainNormalizer(method='vahadane')
    # normalizer.fit(reference_image)
    
    main()