import pandas as pd
import numpy as np
import cv2
#import staintools
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
from sklearn.model_selection import cross_val_score
#from flaml import AutoML
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import roc_auc_score
#from sklearn.decomposition import PCA


def main():

    features = []
    #i=0
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc='Extracting features'):
        image_path = './train/' + row['name'] + '.jpg'
        
        resized_image_array = utils.preprocess_image(image_path) #preprocessing 
        hsv_histogram = utils.color_feature_extractor(resized_image_array)
        #each bin as separate feature
        hsv_features = {f'hsv_bin_{i}': hsv_histogram[i] for i in range(len(hsv_histogram))} 

        gray_image = cv2.cvtColor(resized_image_array, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 3)     

        glcm_features = utils.glcm_feature_extractor(gray_image) #feature extraction
        hu_moments = utils.hu_invariant_moments(gray_image)
        hu_moments = {f'hu_moments_{i}': hu_moments[i] for i in range(len(hu_moments))}

        combined_features = {**hsv_features, **glcm_features, **hu_moments} 
        #print(combined_features) 
        one_part = {
            'image_id': row['name'],
            'label': row['label'],
            **combined_features
        }
        features.append(one_part)

    features_dataframe = pd.DataFrame(features) #overall feature set
    #features_dataframe.to_csv('current_feature_set.csv', index=False)

    X_train, X_test, y_train, y_test = train_test_split(features_dataframe.drop(['image_id', 'label'], axis=1), features_dataframe['label'], test_size=0.2, random_state=42)

    
    #pca = PCA(n_components=0.95)

    #to handle class imbalance
    # smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
    # x_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    # X_train = x_train_smote
    # y_train = y_train_smote
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y = y_train

    X_test_scaled = scaler.transform(X_test)

    #uncomment this when you use PCA
    #X_train_scaled = pca.fit_transform(X_train_scaled)
    #X_test_scaled = pca.transform(X_test_scaled)

    clf_1 = SVC(kernel='linear', verbose = True, C=100, decision_function_shape='ovo', class_weight='balanced', random_state=42)
    clf_2 = XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=len(np.unique(y)), max_depth=6)
    clf_3 = lightgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), random_state=42, num_leaves=31, force_col_wise=True, metric='multi_logloss')
    clf_4 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', criterion='gini')
    
    eclf = VotingClassifier(estimators=[('svc', clf_1), ('xgb', clf_2), ('lgbm', clf_3), ('rf', clf_4)], voting='hard')
    eclf.fit(X_train_scaled, y)

    #to train on separate models
    # eclf = clf_3                #lgbm
    # eclf.fit(X_train_scaled, y)

    #METRICS

    train_predictions = eclf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_true=y, y_pred=train_predictions)
    print("Training accuracy: ", train_accuracy)

    test_predictions = eclf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_true=y_test, y_pred=test_predictions)
    print("Test accuracy: ", test_accuracy)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=test_predictions)
    print("Confusion matrix: \n", conf_matrix)

    clf_report = classification_report(y_true=y_test, y_pred=test_predictions)
    print("Classification report: \n", clf_report)

    #save model
    #joblib.dump(eclf, 'color_hist_glcm_525f_try.pkl')    

if __name__ == '__main__':

    CSV_PATH = 'train.csv'
    IMG_DIR = './train'
    dataframe = pd.read_csv(CSV_PATH)
    main()