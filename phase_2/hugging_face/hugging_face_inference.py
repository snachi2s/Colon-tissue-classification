import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm


TEST_IMG_DIR = "test"  
CSV_FILE = "test.csv"

df = pd.read_csv(CSV_FILE)

#---------------------------------
# Load the trained model from Hub
#---------------------------------
model_name = "selvaa/vit-colon-cancer-classification"   #repo is public
classifier = pipeline(model=model_name)



#-----------------------------------
# structure needed for submission
#-----------------------------------
class_mapping = {
                'Normal_Tissue': 0,
                'Serrated_Lesion': 1,
                'Adenocarcinoma': 2,
                'Adenoma': 3
                }

#-----------------------
# model prediction
#-----------------------
def predict_hf(image_path, classifier):
    outputs = classifier(image_path)
    max_class = max(outputs, key=lambda x: x['score'])
    predicted_class = max_class['label'] # HF label format{0: 'Adenocarcinoma', 1: 'Adenoma', 2: 'Normal_Tissue', 3: 'Serrated_Lesion'}
    return predicted_class

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['name']
    image_path = os.path.join(TEST_IMG_DIR, image_name + '.jpg')
    predicted_class = predict_hf(image_path, classifier) #predict class name 
    predicted_class_id = class_mapping[predicted_class] #mapping HF -> our structure
    df.at[index, 'label'] = predicted_class_id

df['label'] = df['label'].astype(int)
df.to_csv('submission.csv', index=False)