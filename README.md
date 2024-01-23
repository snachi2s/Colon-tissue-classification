# GROUP 5
# Colon-cancer-classification

# PHASE 2: DL


## Hugging Face Hub

- Our best model is loaded at [selvaa/vit-colon-cancer-classification](https://huggingface.co/selvaa/vit-colon-cancer-classification)

## Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| ResNet50            | 76(%)    | 75(%)     | 76(%)  | 75(%)    |
| timm-efficientnetb3 | 81(%)    | 81(%)     | 81(%)  | 81(%)    |
| YOLO-v8x            | 80(%)    | 79(%)     | 78(%)  | 78(%)    |
| vit-base-patch16-224| 83(%)    | 83(%)     | 83(%)  | 82(%)    |

## Folder structure

Inside phase_2,

- **hugging_face_training.py** - leverages Hugging Face library to train, validate and pushning model to HF Hub. Current pipeline supports google/vit  and facebook/deit variants
- **hugging_face_inference.py** - loads trained models from HF Hub and does inference on that --> creates predictions csv too
- **custom_pipeline/** - contains the training, utils and inference scripts to train  torchvision & timm models
- **pytorch_lightning/** - uses PyTorch lightning library for easy training, validation, and logging
- **report.md** - our top3 submission results and their corresponding classification reports

## Methodology

![image info](phase_1/analysis/results/phase2.jpeg)

For training it using hugging face,

```python
python phase_2/hugging_face_train.py 
#similarly for lightning and custom pipeline
```

# PHASE 1: ML CLASSIFIERS

## Best Performing

## Features & Results
| Feature extraction methods     | Computed features                                                                |
|--------------------------------|----------------------------------------------------------------------------------|
| First order statistical features | Mean, Standard deviation, Median, Percentile 25%, Percentile 50%, Percentile 75% |
| GLCM                           | Contrast, Dissimilarity, Homogeneity, Energy, ASM, Correlation, Entropy, Variance|
| Histogram                      | Histogram of bins=8 (in HSV image space)                                         |
| Hu invariant moments           | h1, h2, h3, h4, h5, h6, h7                                                       |

## Results

| Model                                          | Accuracy | Precision | Recall | F1-Score |
|------------------------------------------------|----------|-----------|--------|----------|
| Voting-based classifier (SVM, XGBoost, LGBM, RF)| 80.4(%)  | 77(%)     | 76(%)  | 76(%)    |
| LightGBM                                       | 79.2(%)  | 76(%)     | 75(%)  | 76(%)    |

## Folder structure

Inside phase_1,

- **previous_best** - this folder contains the feature extraction and training pipeline for our previous best approach 
- **analysis** - folder contains scripts for t-SNE and PCA analysis for feature set visualization 
- **analysis\results** - results contains the images that represents the histogram analysis of RGB channels for each type of cancer 
- **utils.py** - feature extraction methods
- **train.py** - training pipeline for voting based classifier
- **submission_pipeline.py** - trains and saves the predictions in the submission format

## Methodology 

![image info](./phase_1/analysis/results/image.png)

For normal training and checking metrics, 
```python
python phase_1/train.py
```

To train the model and generate the predictions CSV file, run
```python
python phase_1/submission_pipeline.py
```

## List of feature extraction methods (we tried):

- GLCM
- Hu invariant moments
- First-order statistical features
- Color histogram (RGB, HSV, LAB)
- LBP
- Gabor filter
- Discrete Fourier Transform (mean, median, std_dev, energy from magnitude spectrum)
- Circular hough transform



