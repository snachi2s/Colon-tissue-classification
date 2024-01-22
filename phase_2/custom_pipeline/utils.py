import torch
from tqdm import tqdm
#import time
#import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#---------------------

def train(trainloader, model, optimizer, scheduler, loss_fn, DEVICE='cuda'):
    model.train()
    loop = tqdm(trainloader, leave=True)
    train_losses = []

    for batch_idx, (image, targets) in enumerate(loop):
        image = image.to(DEVICE) #[batch_size, 3, 224, 224]
        #print("image.shape: ", image.shape)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        preds = model(image) 
        #loss_1 = loss_fn(preds, targets)
        #loss_2 = torch.nn.functional.cross_entropy(preds, targets)
        #loss = loss_1 + loss_2
        loss = loss_fn(preds, targets)
        train_losses.append(loss.item()) #loss over all batches
        loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()
        print("INFO: Learning rate: ", scheduler.get_last_lr()[0])
        return np.mean(train_losses), scheduler.get_last_lr()[0]
    else:
        return np.mean(train_losses), None

#----------------------
# Validation
#----------------------
def validation(validloader, model, loss_fn, epoch, DEVICE='cuda'):
    model.eval()
    validation_losses = []

    all_preds = []
    all_targets = []

    loop = tqdm(validloader, leave=True)
    with torch.no_grad():
        for batch_idx, (image, targets) in enumerate(loop):
            image = image.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(image)
            #loss_1 = loss_fn(preds, targets)
            #loss_2 = torch.nn.functional.cross_entropy(preds, targets)
            #loss = loss_1 + loss_2
            loss = loss_fn(preds, targets)
            validation_losses.append(loss.item())

            all_preds += preds.argmax(dim=1).cpu().tolist()
            all_targets += targets.cpu().tolist()
    
    print("Classification Report:\n", classification_report(all_targets, all_preds, zero_division=1))
    print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_preds, zero_division=1)

    for i in range(len(precision)):
        wandb.log({
            'epoch': epoch,
            f'class_{i}/precision': precision[i],
            f'class_{i}/recall': recall[i],
            f'class_{i}/f1_score': f1_score[i]
        })

    avg_f1_score = sum(f1_score)/len(f1_score)

    return np.mean(validation_losses), avg_f1_score


def get_class_counts(trainset, validset): #for weighted sampler use
    train_counts = [0, 0, 0, 0]
    valid_counts = [0, 0, 0, 0]
    for _, label in trainset:
        train_counts[label] += 1
    for _, label in validset:
        valid_counts[label] += 1
    return train_counts, valid_counts


#---------------------
# Dice loss
#---------------------
def one_hot_encode(targets, num_classes):
    one_hot = torch.zeros(targets.size(0), num_classes, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    return one_hot


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights #from sklearn

    def forward(self, inputs, targets, smooth=1e-6):
        targets_one_hot = one_hot_encode(targets, inputs.size(1)) #1channel -> 4 channel
        inputs = F.softmax(inputs, dim=1) #inpus -> probabilities

        total_loss = 0
        for i in range(targets_one_hot.shape[1]): #items in batch
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)

            intersection = (input_flat * target_flat).sum()
            dice_loss = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
            class_loss = 1 - dice_loss
            weighted_loss = self.weights[i] * class_loss 
            total_loss += weighted_loss

        return total_loss / targets_one_hot.shape[1]
    
