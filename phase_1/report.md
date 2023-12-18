# Features: GLCM, Hue invariant moments, color histogram (COMMON)

## model: LGBM alone
## without using SMOTE & PCA

```python
[LibSVM][LightGBM] [Info] Total Bins 39074
[LightGBM] [Info] Number of data points in the train set: 4532, number of used features: 219
Training accuracy:  1.0
Test accuracy:  0.7925465838509317
Confusion matrix:  [[213  24   4   7]
 [ 24 102  16  25]
 [ 11   6 249   5]
 [ 12  26   7  74]]
Classification report:                precision    recall  f1-score   support

                                0       0.82      0.86      0.84       248
                                1       0.65      0.61      0.63       167
                                2       0.90      0.92      0.91       271
                                3       0.67      0.62      0.64       119

                            accuracy                        0.79       805
                        macro avg       0.76      0.75      0.76       805
                        weighted avg    0.79      0.79      0.79       805
```



## model: Voting based classifier (SVM, XGB, LGBM, RF)

## without using SMOTE & PCA

```python
[LibSVM][LightGBM] [Info] Total Bins 34173
[LightGBM] [Info] Number of data points in the train set: 3216, number of used features: 191
[LightGBM] [Info] Start training from score -1.315479
[LightGBM] [Info] Start training from score -1.561181
[LightGBM] [Info] Start training from score -1.043269
[LightGBM] [Info] Start training from score -1.775108
Training accuracy:  1.0
Test accuracy:  0.8049689440993789
Confusion matrix:  
[[227  13   3   5]
 [ 29 101  15  22]
 [ 12   7 249   3]
 [ 13  27   8  71]]
Classification report:                precision    recall  f1-score   support

                                0       0.81      0.92      0.86       248
                                1       0.68      0.60      0.64       167
                                2       0.91      0.92      0.91       271
                                3       0.70      0.60      0.65       119

                        accuracy                           0.80       805
                       macro avg       0.77      0.76      0.76       805
                    weighted avg       0.80      0.80      0.80       805
```

## without PCA & with SMOTE

```python

[LightGBM] [Info] Number of data points in the train set: 4532, number of used features: 219
[LightGBM] [Info] Start training from score -1.386294
[LightGBM] [Info] Start training from score -1.386294
[LightGBM] [Info] Start training from score -1.386294
[LightGBM] [Info] Start training from score -1.386294
Training accuracy:  1.0
Test accuracy:  0.7900621118012422
Confusion matrix:  [[221  18   4   5]
 [ 27 104  12  24]
 [ 15   8 243   5]
 [ 14  27  10  68]]
Classification report:                precision    recall  f1-score   support

                                 0       0.80      0.89      0.84       248
                                 1       0.66      0.62      0.64       167
                                 2       0.90      0.90      0.90       271
                                 3       0.67      0.57      0.62       119

                           accuracy                           0.79       805
                          macro avg       0.76      0.75      0.75       805
                       weighted avg       0.79      0.79      0.79       805
```
