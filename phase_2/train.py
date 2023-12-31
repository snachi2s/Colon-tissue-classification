import torch
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from dataset import ColonCanerDataset
from tqdm import tqdm
from torchvision.models.resnet import ResNet50_Weights
import utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
#TEST_BATCH_SIZE = 4

EPOCHS = 30
NUM_WORKERS = 4
PIN_MEMORY = True
TRAIN_IMG_DIR = "train"
TRAIN_CSV = "train.csv"
NUM_CLASSES = 4

LOAD_MODEL = False

# Augmentation parameters
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def main():

    #augmentations
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #using imagenet mean and std
            ToTensorV2(),
        ]
    )

    if transform is not None:
        print("INFO: Augmentations applied")

    #load the images and labels 
    dataset = ColonCanerDataset(
        image_dir=TRAIN_IMG_DIR,
        annotation_file=TRAIN_CSV,
        transform=transform,
    )

    #split: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    trainset, validset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    print(f"INFO: Training data split, TRAIN: {train_size}, VALID: {valid_size}")

    #test_size = len(dataset) - train_size - valid_size
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    # print("INFO: training data split, train: {train_size}, valid: {valid_size}, test: {test_size}")

    #dataloaders
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=validset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=TEST_BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=False,
    # )

    #loading model and setting hyperparameters
    #using pretrained ResNet50 pre-trained model
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) 
    model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES) #changing last layer 
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = 0.90
    best_epoch = 0
    print("INFO: Training started")

    for epoch in range(1, EPOCHS+1):
        print(f"Epoch: {epoch}/{EPOCHS}")

        train_loss = utils.train(train_loader, model, optimizer, loss_fn)
        valid_loss = utils.validation(valid_loader, model, loss_fn)

        print(f"INFO: Training loss: {train_loss:.3f}")
        print(f"INFO: Validation loss {valid_loss:.3f}")

        #saving best model based on losses
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "best_resnet50_model.pth")
            best_epoch = epoch
            print(f"Best model saved at epoch {best_epoch} with loss: {best_loss:.3f}")
    print("INFO: Training completed")

if __name__ == "__main__":
    main()