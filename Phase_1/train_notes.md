## For using voting based classifier

```python
clf_1 = SVC(kernel='linear', verbose = True, C=100, decision_function_shape='ovo', random_state=42)
clf_2 = XGBClassifier(booster='gbtree', objective='multi:softmax', num_class=len(np.unique(y)), max_depth=6)
clf_3 = lightgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), random_state=42, num_leaves=31)
clf_4 = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='adam', random_state=42)
clf_4.out_activation_ = 'softmax'
    
eclf = VotingClassifier(estimators=[('svm', clf_1), ('xgb', clf_2), ('lgbm', clf_3), ('mlp', clf_4)], voting='hard')
eclf.fit(X_train_scaled, y)
```

## Using grid search for finding the hyperparameters

```python
#define the hyperparameters with possible guesses
param_grid = {
        'num_leaves': [35, 65],
        'reg_alpha': [0.1, 0.5],
        'min_data_in_leaf': [30, 50, 100],
        'lambda_l1': [0.5, 1],
         lambda_l2': [0.5, 1]
    }

lgb_estimator = lightgbm.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=len(np.unique(y)), random_state=42)
grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid)
grid_search.fit(x_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
```

## List of feature extraction methods (we tried):

- GLCM
- Hu invariant moments
- First order statistical features
- Color histogram
- LBP
