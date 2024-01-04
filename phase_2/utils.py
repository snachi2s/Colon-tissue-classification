import torch
from tqdm import tqdm
#import time
#import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix

def train(trainloader, model, optimizer, scheduler, loss_fn, DEVICE='cuda'):
    loop = tqdm(trainloader, leave=True)
    running_loss = 0.0

    model.train()
    for batch_idx, (image, targets) in enumerate(loop):
        image = image.to(DEVICE) #[batch_size, 3, 224, 224]
        #print("image.shape: ", image.shape)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        preds = model(image) 
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item()) #loss over items in a single batch
        running_loss += loss.item() #loss over all batches

    if scheduler is not None:
        scheduler.step()
        return running_loss/len(trainloader), scheduler.get_last_lr()[0]
    else:
        return running_loss/len(trainloader), None


def validation(validloader, model, loss_fn, DEVICE='cuda'):
    model.eval()
    valid_loss = 0.0

    all_preds = []
    all_targets = []

    loop = tqdm(validloader, leave=True)
    with torch.no_grad():
        for batch_idx, (image, targets) in enumerate(loop):
            image = image.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(image)
            loss = loss_fn(preds, targets)
            loop.set_postfix(loss=loss.item())
            valid_loss += loss.item()

            all_preds += preds.argmax(dim=1).cpu().tolist()
            all_targets += targets.cpu().tolist()

    print("Classification Report:\n", classification_report(all_targets, all_preds, zero_division=1))
    print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))

    return valid_loss/len(validloader)