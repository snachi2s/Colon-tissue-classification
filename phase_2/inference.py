import pandas as pd
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMG_DIR = "test"  
MODEL_WEIGHTS_PATH = 'trained_models/best_resnet50_epoch_12_model.pth'
CSV_FILE = "test.csv"

model = timm.create_model('resnet50', pretrained=False, num_classes=4)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.to(DEVICE)
model.eval()
print("INFO: MODEL LOADED !!!")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) #add batch dimension
    image = image.to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = probabilities.argmax().item()
    return int(predicted_class)

df = pd.read_csv(CSV_FILE)

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['name']
    image_path = os.path.join(TEST_IMG_DIR, image_name + '.jpg')
    predicted_class = predict(image_path, model, transform)
    df.at[index, 'label'] = predicted_class

df['label'] = df['label'].astype(int)
df.to_csv('predictions.csv', index=False)
