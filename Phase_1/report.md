
# Changing models

## With color_hist(RGB, HSV, LAB), Hu, Statistical, GLCM

```python
clf_1 = SVC(kernel='linear', verbose = True, C=100, decision_function_shape='ovo' class_weight='balanced', random_state=42)
clf_2 = XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=len(np.unique(y)), max_depth=6)
clf_3 = lightgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), random_state=42, num_leaves=31)
clf_4 = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', random_state=42)
clf_4.out_activation_ = 'softmax'

eclf = VotingClassifier(estimators=[('svm', clf_1), ('xgb', clf_2), ('lgbm', clf_3), ('mlp', clf_4)], voting='hard')
eclf.fit(X_train_scaled, y)
```

```
Training accuracy:  0.9959577114427861
Test accuracy:  0.7316770186335404

Confusion matrix: 
[[212  20  11   5]
 [ 35  76  31  25]
 [  9  18 239   5]
 [ 12  26  19  62]]

Classification report: 
              precision    recall  f1-score   support

           0       0.79      0.85      0.82       248
           1       0.54      0.46      0.50       167
           2       0.80      0.88      0.84       271
           3       0.64      0.52      0.57       119

    accuracy                           0.73       805
   macro avg       0.69      0.68      0.68       805
weighted avg       0.72      0.73      0.72       805
```

## same features without MLP classifier

```
Training accuracy:  1.0
Test accuracy:  0.7229813664596273
Confusion matrix: 
[[206  19  18   5]
 [ 34  72  31  30]
 [ 11  13 242   5]
 [ 15  22  20  62]]
 
Classification report: 
              precision    recall  f1-score   support

           0       0.77      0.83      0.80       248
           1       0.57      0.43      0.49       167
           2       0.78      0.89      0.83       271
           3       0.61      0.52      0.56       119

    accuracy                           0.72       805
   macro avg       0.68      0.67      0.67       805
weighted avg       0.71      0.72      0.71       805
```

## same features without MLP and XGB

```
Training accuracy:  0.8439054726368159
Test accuracy:  0.715527950310559
Confusion matrix: 
[[218  23   6   1]
 [ 43  85  20  19]
 [ 20  25 225   1]
 [ 17  34  20  48]]
Classification report: 
              precision    recall  f1-score   support

           0       0.73      0.88      0.80       248
           1       0.51      0.51      0.51       167
           2       0.83      0.83      0.83       271
           3       0.70      0.40      0.51       119

    accuracy                           0.72       805
   macro avg       0.69      0.66      0.66       805
weighted avg       0.71      0.72      0.71       805
```

## only with svm

```
[LibSVM]Training accuracy:  0.7080223880597015
Test accuracy:  0.675776397515528
Confusion matrix: 
[[184  40  16   8]
 [ 33  64  22  48]
 [ 14  15 231  11]
 [ 10  27  17  65]]
Classification report: 
              precision    recall  f1-score   support

           0       0.76      0.74      0.75       248
           1       0.44      0.38      0.41       167
           2       0.81      0.85      0.83       271
           3       0.49      0.55      0.52       119

    accuracy                           0.68       805
   macro avg       0.63      0.63      0.63       805
weighted avg       0.67      0.68      0.67       805
```

# Changing the feature set

## Voting based classifier without Hu moments
```
[LightGBM] [Info] Total Bins 14844
[LightGBM] [Info] Number of data points in the train set: 3216, number of used features: 59
[LightGBM] [Info] Start training from score -1.315479
[LightGBM] [Info] Start training from score -1.561181
[LightGBM] [Info] Start training from score -1.043269
[LightGBM] [Info] Start training from score -1.775108
Training accuracy:  0.9735696517412935
Test accuracy:  0.7490683229813665
Confusion matrix: 
[[215  18  10   5]
 [ 36  82  30  19]
 [ 11  16 242   2]
 [ 12  24  19  64]]
Classification report: 
              precision    recall  f1-score   support

           0       0.78      0.87      0.82       248
           1       0.59      0.49      0.53       167
           2       0.80      0.89      0.85       271
           3       0.71      0.54      0.61       119

    accuracy                           0.75       805
   macro avg       0.72      0.70      0.70       805
weighted avg       0.74      0.75      0.74       805
```

## without hu & glcm

```
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 9744
[LightGBM] [Info] Number of data points in the train set: 3216, number of used features: 39
[LightGBM] [Info] Start training from score -1.315479
[LightGBM] [Info] Start training from score -1.561181
[LightGBM] [Info] Start training from score -1.043269
[LightGBM] [Info] Start training from score -1.775108
Training accuracy:  0.9732587064676617
Test accuracy:  0.737888198757764
Confusion matrix: 
[[210  16  15   7]
 [ 36  85  25  21]
 [ 14  13 243   1]
 [ 15  28  20  56]]

Classification report: 
              precision    recall  f1-score   support

           0       0.76      0.85      0.80       248
           1       0.60      0.51      0.55       167
           2       0.80      0.90      0.85       271
           3       0.66      0.47      0.55       119

    accuracy                           0.74       805
   macro avg       0.71      0.68      0.69       805
weighted avg       0.73      0.74      0.73       805
```

## only with histogram features (RGB, HSV, LAB)
```
[LightGBM] [Info] Total Bins 8469
[LightGBM] [Info] Number of data points in the train set: 3216, number of used features: 34
[LightGBM] [Info] Start training from score -1.315479
[LightGBM] [Info] Start training from score -1.561181
[LightGBM] [Info] Start training from score -1.043269
[LightGBM] [Info] Start training from score -1.775108
Training accuracy:  0.9751243781094527
Test accuracy:  0.6658385093167701
Confusion matrix: 
[[192  16  36   4]
 [ 38  62  32  35]
 [ 15  19 232   5]
 [ 15  32  22  50]]
Classification report: 
              precision    recall  f1-score   support

           0       0.74      0.77      0.76       248
           1       0.48      0.37      0.42       167
           2       0.72      0.86      0.78       271
           3       0.53      0.42      0.47       119

    accuracy                           0.67       805
   macro avg       0.62      0.61      0.61       805
weighted avg       0.65      0.67      0.65       805
```


