from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
 
class ColonCanerDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.csv = pd.read_csv(annotation_file)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):  #iterator over the dataset
        """
        index: acts as an iterator over the dataset

        return:
        image: torch tensor of format [batch_size, height, width, channels]
        label: torch tensor of integer type of format [batch_size, label_value]
        """

        image_path = os.path.join(self.image_dir, self.images[index])
        #print("image_path: ", image_path)

        #read image and labels
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_basename = os.path.splitext(os.path.basename(image_path))[0] #without '.jpg' 
        #print("image_basename: ", image_basename)

        #check the basename present in the name column in csv file and get the corresponding label value
        label = self.csv[self.csv['name'] == image_basename]['label'].values[0]
        #print("label: ", label)

        #applying augmentations
        if self.transform:  
            augmentations = self.transform(image=image)
            image = augmentations['image']

        return image, label
    

#sanity check
# if __name__ == "__main__":
    
#     dataset = ColonCanerDataset(
#         image_dir="train",
#         annotation_file="train.csv",
#         transform=None,
#     )

#     train_dataloader = DataLoader(
#         dataset = dataset,
#         batch_size=2,
#         num_workers=2,
#         pin_memory=True,
#         shuffle=True,
#     )

#     for _, (image, label) in enumerate(train_dataloader):
#         print("image.shape: ", image.shape)
#         print("label.shape: ", label.shape)
#         print("label: ", label)
#         break
