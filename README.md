# GROUP 5
# Colon-tissue-classification

# PHASE 2: DL

## Folder structure

Inside phase_2,

- **hugging_face_training.py** - leverages Hugging Face library to train, validate and pushning model to HF Hub. Current pipeline supports google/vit  and facebook/deit variants
- **hugging_face_inference.py** - loads trained models from HF Hub and does inference on that --> creates predictions csv too
- **custom_pipeline/** - contains the training, utils and inference scripts to train  torchvision & timm models
- **pytorch_lightning/** - uses PyTorch lightning library for easy training, validation, and logging
- **report.md** - our top3 submission results and their corresponding classification reports

## Methodology

![image info](phase_1/analysis/results/phase2.jpeg)

For training it in hugging face,

```python
python phase_2/hugging_face_train.py 
#similarly for lightning and custom pipeline
```

## Hugging Face Hub

- Our best model is loaded at [selvaa/vit-colon-cancer-classification](https://huggingface.co/selvaa/vit-colon-cancer-classification)

# PHASE 1: ML 

## List of feature extraction methods (we tried):

- GLCM
- Hu invariant moments
- First-order statistical features
- Color histogram (RGB, HSV, LAB)
- LBP
- Gabor filter
- Discrete Fourier Transform (mean, median, std_dev, energy from magnitude spectrum)
- Circular hough transform

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



