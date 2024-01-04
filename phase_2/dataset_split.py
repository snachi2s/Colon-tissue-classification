'''
Function to split the images in the given train folder into train, valid, and test sets
'''

import os
import shutil
from sklearn.model_selection import train_test_split

original_train_folder = 'train'

all_images = []

for file in os.listdir(original_train_folder):
    if os.path.isfile(os.path.join(original_train_folder, file)):
        all_images.append(file)

# Split the images into train, valid, and test sets [70%, 15%, 15%]
train_files, test_val_files = train_test_split(all_images, test_size=0.3, random_state=42)  
valid_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

def create_and_move(files, folder_name):
    '''
    Creates a folder 'split_dataset' in the current directory and moves the files into the respective folders
    '''
    dest_folder = os.path.join('split_dataset/', folder_name)
    os.makedirs(dest_folder, exist_ok=True)
    for file in files:
        shutil.move(os.path.join(original_train_folder, file), os.path.join(dest_folder, file))

create_and_move(train_files, 'train')
create_and_move(valid_files, 'valid')
create_and_move(test_files, 'test')