import torch
#import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from dataset import ColonCanerDataset
import utils
import wandb
import hyperparameters
import timm
import wandb
from sklearn.utils import class_weight
import numpy as np

# wandb Hyperparameter tuning

# sweep_configuration = {
#     "method": "random",
#     "name": "sweep",
#     "metric": {
#                 #"goal": "maximize", "name": "f1_score",
#                 "goal": "minimize", "name": "valid_loss"},
#     "parameters": {
#         "batch_size": {"values": [6, 8, 10]},
#         "lr": {"max": 0.01, "min": 0.0001},
#         },
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweeps-efficientnet_b2a", entity="selvaa")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------
# Hyperparameters
#---------------------

model_name = 'efficientnet_b3a'

LEARNING_RATE = 0.0006
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

EPOCHS = 60
NUM_WORKERS = 2
PIN_MEMORY = True
TRAIN_IMG_DIR = "train"
#VALID_IMG_DIR = "split_dataset/valid"
#TEST_IMG_DIR = "split_dataset/test"
LABEL_CSV = "train.csv"
NUM_CLASSES = 4

EARLY_STOPPING = True
patience = 5

LR_SCHEDULER = False
scheduler_name = 'one_cycle_lr' #available: step_lr, cosine_annealing, multi_step_lr, cyclic_lr, one_cycle_lr

#Augmentation parameters
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

#---------------------
# Main
#---------------------
def main():

    #initializing wandb
    wandb.init(project="colon cancer classification - latest", entity="selvaa")
    config = wandb.config
    # TRAIN_BATCH_SIZE = config.batch_size
    # VALID_BATCH_SIZE = config.batch_size
    # LEARNING_RATE = config.lr

    wandb.config.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    wandb.config.VALID_BATCH_SIZE = VALID_BATCH_SIZE
    wandb.config.LEARNING_RATE = LEARNING_RATE
    wandb.config.epochs = EPOCHS
    wandb.config.NUM_CLASSES = NUM_CLASSES
    wandb.config.LR_SCHEDULER = LR_SCHEDULER

    if LR_SCHEDULER:
        wandb.config.scheduler_name = scheduler_name

    #run_name
    run_name = f"{model_name}_{EPOCHS}epochs_batch_size{TRAIN_BATCH_SIZE}_{LEARNING_RATE}_lr_with_weights_augmentations-flips"
    wandb.run.name = run_name

    #----------------
    #Augmentations
    #----------------
    if model_name == 'vit_base_patch16_224':
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]), #using model pretrained configs
            ToTensorV2(),
        ])

    else:
        transform = A.Compose([
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), #using imagenet mean and std
                ToTensorV2(),
                
            ])

    test_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]), #using imagenet mean and std
            ToTensorV2(),
        ])

    if transform is not None:
        print("INFO: Augmentations applied")
        wandb.config.update({"Augmentations": transform})
        #wandb.config.update({"Test Augmentations": test_transform})

    # -----------------
    # DATALOADERS
    # -----------------
    dataset = ColonCanerDataset(
        image_dir=TRAIN_IMG_DIR,
        annotation_file=LABEL_CSV,
        transform=transform,
    )

    #80-20 split
    valid_size = int(0.2 * len(dataset))
    train_size = len(dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    print(f"INFO: Training samples: {len(train_set)}, Validation samples: {len(valid_set)}")


    #------------------------------------------------
    # uncomment these lines to use weighted sampler
    #------------------------------------------------

    # train_counts, valid_counts = utils.get_class_counts(train_set, valid_set)
    # #print(f"INFO: Training class counts: {train_counts}, Validation class counts: {valid_counts}")

    # class_weights = 1.0 / torch.tensor(train_counts, dtype=torch.float)
    # print(f"INFO: Class weights: {class_weights}")

    # train_labels = [label for _, label in train_set]
    # train_weights = [class_weights[label] for label in train_labels]

#     train_sampler = WeightedRandomSampler(
#         weights=train_weights,
#         num_samples=len(train_weights),
#         replacement=True
#    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        #sampler=train_sampler,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # --------------------------------
    # Model, optimizer, and losses
    # --------------------------------

    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    optimizer = hyperparameters.choose_optimizer(model, optimizer_name='adam', lr=LEARNING_RATE)
    #get labels from trainloader
    y = torch.tensor([label for _, label in train_loader.dataset])
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
    weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE) #to handle class imbalance giving more weights to class 1 and 3
    print(f"INFO: Class weights: {weights}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights) #weight=weights
    #loss_fn = utils.WeightedDiceLoss(weights=weights)
    #loss_fn = utils.FocalLoss(alpha=weights, gamma=2.0)

    if EARLY_STOPPING:
        early_stop = hyperparameters.EarlyStopping(patience=patience, min_delta=0.001)
        print("INFO: Early stopping enabled")

    if LR_SCHEDULER:
        scheduler = hyperparameters.learning_rate_scheduler(optimizer, scheduler_name=scheduler_name, step_size=200, gamma=0.5, min_lr=1e-4)
        print("INFO: Learning rate scheduler enabled")

    else:
        scheduler = None

    ## Training loop
    best_loss = 0.90
    best_score = 0.0
    best_epoch = 0
    print("INFO: Training started")
    for epoch in range(1, EPOCHS+1):
        print(f"Epoch: {epoch}/{EPOCHS}")

        train_loss, lr = utils.train(train_loader, model, optimizer, scheduler, loss_fn)
        valid_loss, f1_score = utils.validation(valid_loader, model, loss_fn, epoch)

        print(f"INFO: Training loss: {train_loss:.3f}")
        print(f"INFO: Validation loss {valid_loss:.3f}")

        # ----------------------------------
        # Early Stopping and wandb logging
        # ----------------------------------
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Valid Loss": valid_loss,
            "f1_score": f1_score,
            })

        if lr is not None:
            wandb.log({"Learning Rate": lr})
        
        #check early stopping every 3 epochs
        if epoch % 2 == 0:
            print("INFO: Checking early stopping")
            if EARLY_STOPPING and early_stop(valid_loss):
                print("INFO: Early stopping")
                break

        # ------------------------------------
        # Saving best model based on f1 score
        # ------------------------------------
        if f1_score > best_score:
            best_score = f1_score
            best_epoch = epoch
            best_state_dict = model.state_dict()

    print("INFO: Training completed")
    saved_model_path = f"trained_models/best_{model_name}_epoch_{best_epoch}_model.pth"
    torch.save(best_state_dict, saved_model_path)
    print(f"INFO: Best model saved at {best_epoch} epoch with loss {best_loss:.3f}")

    wandb.finish()

#wandb.agent(sweep_id, function=main, count=10)

if __name__ == "__main__":
    main()