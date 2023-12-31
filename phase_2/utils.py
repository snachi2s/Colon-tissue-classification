import torch
from tqdm import tqdm
#import time


def train(trainloader, model, optimizer, loss_fn, DEVICE='cuda'):
    loop = tqdm(trainloader, leave=True)
    running_loss = 0.0

    model.train()
    for batch_idx, (image, targets) in enumerate(loop):
        image = image.to(DEVICE) #to GPU
        targets = targets.to(DEVICE)
        #print("data.shape: ", image.shape)
        #print("targets.shape: ", targets.shape)
        #time.sleep(10)

        optimizer.zero_grad()
        predictions = model(image)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item()) #loss over items in a single batch
        running_loss += loss.item() #loss over all batches

    train_loss = running_loss / len(trainloader)
    return train_loss

def validation(validloader, model, loss_fn, DEVICE='cuda'):
    loop = tqdm(validloader, leave=True)
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch_idx, (image, targets) in enumerate(loop):
            image = image.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(image)
            loss = loss_fn(preds, targets)
            loop.set_postfix(loss=loss.item())
            valid_loss += loss.item()

    valid_loss = valid_loss / len(validloader)
    return valid_loss












# def test_fn(test_loader, model, DEVICE='cuda'):
#     test_running_loss=0.0
#     accuracy = 0.0
#     loop = tqdm(test_loader, leave=True)

#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (image, targets) in enumerate(loop):
#             image = image.to(DEVICE)
#             targets = targets.to(DEVICE)

#             preds = model(image)
            
#             predictions = torch.argmax(preds, dim=1) #from probabilities --> get max class 
#             # print("preds: ", predictions)
#             # print("targets: ", targets)

#             loss_fn = torch.nn.CrossEntropyLoss()
#             loss = loss_fn(preds, targets)
#             loop.set_postfix(loss=loss.item())
#             test_running_loss += loss.item()

#         print("test_running_loss: ", test_running_loss)