import torch
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from dataset import ColonCanerDataset
from tqdm import tqdm
from torchvision.models.resnet import ResNet50_Weights
import utils
import wandb   #for visualization, hyperparameter study and for keeping track of runs
import hyperparameters


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
#TEST_BATCH_SIZE = 4

EPOCHS = 35
NUM_WORKERS = 4
PIN_MEMORY = True
TRAIN_IMG_DIR = "split_dataset/train"
VALID_IMG_DIR = "split_dataset/valid"
TEST_IMG_DIR = "split_dataset/test"
LABEL_CSV = "train.csv"
NUM_CLASSES = 4

EARLY_STOPPING = True
LR_SCHEDULER = False

TEST_TRAINED_MODEL = True

# Augmentation parameters
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def main():

    #initializing wandb 
    #use wandb if you're comfortable else use matplotlib
    wandb.init(project="colon tissue classification", entity="selvaa")  #!!!! you need to create your own account
    wandb.config.epochs = EPOCHS
    wandb.config.batch_size = TRAIN_BATCH_SIZE
    wandb.config.learning_rate = LEARNING_RATE
    wandb.config.IMAGE_HEIGHT = IMAGE_HEIGHT
    wandb.config.IMAGE_WIDTH = IMAGE_WIDTH
    wandb.config.NUM_CLASSES = NUM_CLASSES

    #augmentations
    transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #using imagenet mean and std
            ToTensorV2(),
        ])

    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #using imagenet mean and std
            ToTensorV2(),
        ])

    if transform is not None:
        print("INFO: Augmentations applied")
        wandb.config.update({"Augmentations": transform})
        wandb.config.update({"Test Augmentations": test_transform})

    # -----------------
    # DATALOADERS
    # ----------------- 
    train_set = ColonCanerDataset(
        image_dir=TRAIN_IMG_DIR,
        annotation_file=LABEL_CSV,
        transform=transform,
    )

    valid_set = ColonCanerDataset(
        image_dir=VALID_IMG_DIR,
        annotation_file=LABEL_CSV,
        transform=transform,
    )

    test_set = ColonCanerDataset(
        image_dir=TEST_IMG_DIR,
        annotation_file=LABEL_CSV,
        transform=test_transform,
    )

    print(f"INFO: Training samples: {len(train_set)}, Validation samples: {len(valid_set)}, Test samples: {len(test_set)}")
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=valid_set,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # ----------------------------------
    # Model, optimizer, and losses 
    # ----------------------------------
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES)
    #model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    weights = torch.tensor([1.0, 2.0, 1.0,2.0]).to(DEVICE) #to handle class imbalance giving more weights to class 1 and 3
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    if EARLY_STOPPING: 
        early_stop = hyperparameters.EarlyStopping(patience=8, min_delta=0.001, monitor='val_loss')
        print("INFO: Early stopping enabled")

    if LR_SCHEDULER:
        scheduler = hyperparameters.learning_rate_scheduler(optimizer, scheduler_name='step_lr', step_size=4, gamma=0.5, min_lr=1e-5)
        print("INFO: Learning rate scheduler enabled")

    else:
        scheduler = None

    ## Training loop
    best_loss = 0.90
    best_epoch = 0
    print("INFO: Training started")
    for epoch in range(1, EPOCHS+1):
        print(f"Epoch: {epoch}/{EPOCHS}")

        train_loss, lr = utils.train(train_loader, model, optimizer, scheduler, loss_fn)
        valid_loss = utils.validation(valid_loader, model, loss_fn)

        print(f"INFO: Training loss: {train_loss:.3f}")
        print(f"INFO: Validation loss {valid_loss:.3f}")

        # ----------------------------------
        # Early Stopping and wandb logging
        # ----------------------------------
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss, 
            "Valid Loss": valid_loss,
            })
        
        if lr is not None:
            wandb.log({"Learning Rate": lr})
        
        if EARLY_STOPPING and early_stop(val_loss=valid_loss):
            print("INFO: Early stopping")
            break

        # ------------------------------------
        # Saving best model based on val.loss
        # ------------------------------------
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
    
    print("INFO: Training completed")
    saved_model_path = f"trained_models/best_resnet50_epoch_{best_epoch}_model.pth"
    torch.save(best_state_dict, saved_model_path)
    print(f"INFO: Best model saved at {best_epoch} epoch with loss {best_loss:.3f}")

    #testing the trained model
    if TEST_TRAINED_MODEL:
        print("INFO: Testing the trained model")
        model.load_state_dict(torch.load(saved_model_path))
        test_loss = utils.validation(test_loader, model, loss_fn)
        print(f"INFO: Test loss: {test_loss:.3f}")

    #run_name
    run_name = f"resnet50_{EPOCHS}_epochs_{TRAIN_BATCH_SIZE}_batch_size_{LEARNING_RATE}_lr"
    wandb.run.name = run_name

    wandb.finish()

if __name__ == "__main__":
    main()