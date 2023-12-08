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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import utils
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#from flaml import AutoML
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE

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
    #i=0
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Processed Images"):
        image_path = './train/' + row['name'] + '.jpg'
        #print(image_path)
        #time.sleep(8)
        preprocessed_image = utils.preprocess_image(image_path)
        #print(type(preprocessed_image))
        #time.sleep(10)
        mean, median, std_dev, per_25, per_75, glcm_feature_dict, hu_list = feature_extraction(preprocessed_image)
        cv_image = cv2.imread(image_path)
        cv_image = cv2.resize(cv_image, (256, 256))
        hist_RGB_dict = utils.histogram_statistical_features_RGB(cv_image, bins=128)
        hist_HSV_dict = utils.histogram_statistical_features_HSV(cv_image, bins=128)
        hist_LAB_dict = utils.histogram_statistical_features_LAB(cv_image, bins=128)
        
        dft_mean, dft_median, dft_std_dev, dft_energy = utils.extract_dft_features(preprocessed_image)

        one_part = ({
            'image_id': row['name'],
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'percentile_25': per_25,
            'percentile_75': per_75,
            'hu_moment_1': hu_list[0],
            'hu_moment_2': hu_list[1],
            'hu_moment_3': hu_list[2],
            'hu_moment_4': hu_list[3],
            'hu_moment_5': hu_list[4],
            'hu_moment_6': hu_list[5],
            'hu_moment_7': hu_list[6],
            'dft_mean': dft_mean,
            'dft_median': dft_median,
            'dft_std_dev': dft_std_dev,
            'dft_energy': dft_energy,
            'label': row['label'],
        })

        #glcm
        for feature, value in glcm_feature_dict.items():
            for index, value in enumerate(value):
                column_name = f"{feature}{index+1}"
                one_part[column_name] = value

        for feature, value in hist_RGB_dict.items():
            one_part[feature] = value

        for feature, value in hist_HSV_dict.items():
            one_part[feature] = value

        for feature, value in hist_LAB_dict.items():
            one_part[feature] = value

        features.append(one_part)
        
        #sanity check
        # i+=1
        # if i == 10:
        #     break


    features_dataframe = pd.DataFrame(features)
    #features_dataframe.to_csv('analysis/all_features.csv', index=False)
    #print(features_dataframe.head())

    X_train, X_test, y_train, y_test = train_test_split(features_dataframe.drop(['image_id', 'label'], axis=1), features_dataframe['label'], test_size=0.2, random_state=42)

    #to handle class imbalance
    smote = BorderlineSMOTE(random_state=42, kind='borderline-1')

    x_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    X_train = x_train_smote
    y_train = y_train_smote
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y = y_train

    X_test_scaled = scaler.transform(X_test)

    #TODO: why lgbm shows number of features less?
    # pre-processing steps like active contonour method, weiner filter then dft

    clf_1 = SVC(kernel='linear', verbose = True, C=100, decision_function_shape='ovo', class_weight='balanced', random_state=42)
    clf_2 = XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=len(np.unique(y)), max_depth=6)
    clf_3 = lightgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), random_state=42, num_leaves=31)
    clf_4 = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', random_state=42)
    clf_4.out_activation_ = 'softmax'
    
    eclf = VotingClassifier(estimators=[('svm', clf_1), ('xgb', clf_2), ('lgbm', clf_3), ('mlp', clf_4)], voting='hard')
    eclf.fit(X_train_scaled, y)

    #eclf = clf_1
    #eclf.fit(X_train_scaled, y)

    #train metrics
    train_predictions = eclf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_true=y, y_pred=train_predictions)
    print("Training accuracy: ", train_accuracy)

    #test
    test_predictions = eclf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_true=y_test, y_pred=test_predictions)
    print("Test accuracy: ", test_accuracy)

    print("Confusion matrix: ")
    print(confusion_matrix(y_true=y_test, y_pred=test_predictions))

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



###########################################################

##subplots
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(Image.open(image_path))
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# axes[1].imshow(preprocessed_image, cmap='gray')
# axes[1].set_title('Preprocessed Image')
# axes[1].axis('off')

# plt.show()