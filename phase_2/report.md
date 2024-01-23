## resnet50

```python

Epoch: 12/35

Classification Report:
               precision    recall  f1-score   support

           0       0.74      0.89      0.81       228
           1       0.56      0.41      0.47       158
           2       0.94      0.95      0.94       273
           3       0.63      0.61      0.62       145

    accuracy                           0.76       804
   macro avg       0.72      0.71      0.71       804
weighted avg       0.75      0.76      0.75       804

Confusion Matrix:
 [[203  16   2   7]
 [ 48  64   7  39]
 [  4   5 258   6]
 [ 19  29   8  89]]
INFO: Training loss: 0.580
INFO: Validation loss 0.620
Validation loss decreased to 0.62003, resetting patience

```

## 2: timm-eff_net_b3a

```python

classification Report:
               precision    recall  f1-score   support

           0       0.85      0.85      0.85       238
           1       0.71      0.53      0.61       172
           2       0.95      0.96      0.95       274
           3       0.61      0.81      0.70       120

    accuracy                           0.81       804  
   macro avg       0.78      0.79      0.78       804
weighted avg       0.82      0.81      0.81       804  

Confusion Matrix:
 [[203  23   2  10]
 [ 30  91   6  45]
 [  0   5 263   6]
 [  7   9   7  97]]
INFO: Training loss: 0.412
INFO: Validation loss 0.528
Validation loss decreased to 0.52829, resetting patience
```

## vit_base_patch16_224

```python
             precision    recall  f1-score   support

           0       0.94      0.97      0.96       416
           1       0.71      0.77      0.74       189
           2       0.78      0.93      0.85       345
           3       0.81      0.51      0.62       257

    accuracy                           0.83      1207
   macro avg       0.81      0.79      0.79      1207
weighted avg       0.83      0.83      0.82      1207

{'test_loss': 0.615271806716919, 'test_accuracy': 0.8293289146644574, 'test_runtime': 11.6621, 'test_samples_per_second': 103.498, 'test_steps_per_second': 12.948}
```