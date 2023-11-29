from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import numpy as np
import cv2
import pandas as pd
#from scipy import stats
from sklearn.svm import SVC

def plot_unsharped(original, applied, title_original='original', title_applied = 'enhanced'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(original)
    ax[0].set_title(title_original)
    ax[0].axis('off')
    
    ax[1].imshow(applied, cmap=plt.cm.gray)
    ax[1].set_title(title_applied)
    ax[1].axis('off')

    fig.tight_layout()
    plt.show()

image = Image.open(r'C:\Users\DELL\Desktop\ISM\data\Adenocarcinoma\0a371c34-01b8-418c-9467-440cf876248c.jpg')
image_2 = Image.open(r'C:\Users\DELL\Desktop\ISM\data\Normal_Tissue\0a3b9f0e-6a8d-48a1-917a-dd86576969db.jpg')

#data pre-processing
image = np.array(image.resize(size=(200, 200)))
image_2 = np.array(image_2.resize(size=(200, 200)))

unsharped_image = unsharp_mask(image, radius = 2, amount = 5, channel_axis = 2) #channel_axis defaults to gray scale
print('unsharped', unsharped_image.shape)

unsharped_image_2 = unsharp_mask(image_2, radius = 2, amount = 5, channel_axis = 2) #channel_axis defaults to gray scale
print('unsharped', unsharped_image_2.shape)

#gray scale conversion
gray_processed = cv2.cvtColor(np.float32(unsharped_image), cv2.COLOR_BGR2GRAY)
print('gray shape:', gray_processed.shape)

gray_processed_2 = cv2.cvtColor(np.float32(unsharped_image_2), cv2.COLOR_BGR2GRAY)
print('gray shape:', gray_processed_2.shape)

plot_unsharped(original=image, applied=gray_processed)
plot_unsharped(original=image_2, applied=gray_processed_2)

## feature extraction and storing it in pandas

#statistical features
mean, median, std_dev = np.mean(gray_processed), np.median(gray_processed), np.std(gray_processed)
mean_2, median_2, std_dev_2 = np.mean(gray_processed_2), np.median(gray_processed_2), np.std(gray_processed_2)


features = {
    'image_id': ['0a371c34-01b8-418c-9467-440cf876248c', '0a3b9f0e-6a8d-48a1-917a-dd86576969db'],
    'mean': [mean, mean_2],
    'median': [median, median_2],
    'std_dev': [std_dev, std_dev_2],
    'label': [2, 0]
}


dataframe = pd.DataFrame(features)
print(dataframe)

x_train = dataframe[['mean', 'median', 'std_dev']]
print(x_train.shape)
y_train = np.array(dataframe[['label']])
y_train = y_train.ravel() #(n_samples, )
print(y_train.shape)

classifier = SVC()
classifier.fit(x_train, y_train)

model_prediction = classifier.predict(x_train)
print('predicted:', model_prediction)