from dataset import ColonCanerDataset
import torch
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import StepLR
#import wandb
from sklearn.utils import class_weight
import numpy as np

torch.set_float32_matmul_precision('high')

#wandb_logger = WandbLogger(project="LIGHTNING - ISM", entity="selvaa")

#-------------------------------
# Model training & validation
#--------------------------------
class Classifier(pl.LightningModule):
    def __init__(self, model_name, num_classes, learning_rate=1e-4, class_weights=torch.tensor([0.8907, 1.2459, 0.7077, 1.5113])):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        #wandb.log({'train_loss': loss})
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_labels = []
        self.val_preds = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        #wandb.log({'val_loss': loss})
        preds = torch.argmax(logits, dim=1)
        self.val_labels.append(labels)
        self.val_preds.append(preds)
        return loss

    def on_validation_epoch_end(self) -> None:
        labels = torch.cat(self.val_labels).detach().cpu().numpy()
        preds = torch.cat(self.val_preds).detach().cpu().numpy()

        report = classification_report(labels, preds, 
                                       target_names=[f'class_{i}' for i in range(4)], 
                                       output_dict=True, zero_division=0) #returns report structure as dict 
        #print("Classification Report:\n", report)
        self.log_dict({f'val_{k}': v['f1-score'] for k, v in report.items() if k != 'accuracy'}, prog_bar=False)

        #weighted accuracy
        self.log('val_f1score_weighted_avg', report['weighted avg']['f1-score'])
        #wandb.log({'val_f1score_weighted_avg': report['weighted avg']['f1-score']})
        self.current_f1score = report['weighted avg']['f1-score']
        self.current_accuracy = report['accuracy']
        print(f"Epoch {self.current_epoch}: F1 Score: {self.current_f1score:.4f}")
        print(f"Epoch {self.current_epoch}: Accuracy: {self.current_accuracy:.4f}")

        #-----------
        # accuracy
        self.log('val_accuracy', report['accuracy'])
        #wandb.log({'val_accuracy': report['accuracy']})

    def on_epoch_end(self):
        print(f"Epoch {self.current_epoch}: F1 Score: {self.current_f1score:.4f}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=600,  
            epochs=self.trainer.max_epochs,               
            anneal_strategy='linear',
            total_steps=self.trainer.max_epochs * 322                     
        ) #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html

        #using step_lr as scheduler
        #scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=7, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.train_dataset = None
        self.val_dataset = None

    def class_weights(self):
        overall_labels = [label for _, label in self.train_dataset]

        class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                          classes=np.unique(overall_labels), 
                                                          y=overall_labels)
        weights = torch.tensor(class_weights, dtype=torch.float)
        return weights

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = ColonCanerDataset(image_dir=self.data_dir, transform=self.transform)
        train_size = int(0.7 * len(dataset))
        #20 for validation
        val_size = int(0.3 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((260,260)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#--------------------------
# dataloader
#--------------------------
data_module = DataModule(data_dir='/home/ant_devel/Desktop/semester 3/ISM/DL/train', 
                         transform=transform, 
                         batch_size=12)

data_module.setup()
class_weights = data_module.class_weights()
print("INFO: Class weights: ", class_weights)

#model and training
model_name='ens_adv_inception_resnet_v2'
model = Classifier(model_name=model_name, 
                   num_classes=4,
                   learning_rate=0.001,
                   class_weights=class_weights)


checkpoint = ModelCheckpoint(
    monitor='val_f1score_weighted_avg',
    dirpath='checkpoints',
    filename=f'best-{model_name}-checkpoint',
    save_top_k=1,
    mode='max'
)

trainer = Trainer(max_epochs=100, 
                  precision="32", 
                  #logger=wandb_logger, 
                  callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=6, verbose=True),
                             checkpoint],
                  accumulate_grad_batches=3,
                  )

trainer.fit(model, datamodule=data_module)



