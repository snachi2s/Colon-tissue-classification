import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR

#---------------------
# Optimizers
#---------------------
def choose_optimizer(model, optimizer_name, lr):

    optimizers_dict = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
    }
    optimizer = optimizers_dict.get(optimizer_name.lower())

    if optimizer_name.lower() == 'sgd':
        optimizer = optimizer(model.parameters(), lr=lr, momentum=0.9)

    elif optimizer_name.lower() == 'adamw':
        optimizer = optimizer(model.parameters(), lr=lr, weight_decay=1e-6)

    else:
        optimizer = optimizer(model.parameters(), lr=lr) 

    return optimizer 

def learning_rate_scheduler(optimizer, scheduler_name, step_size=120, gamma=0.5, min_lr=1e-5):

    scheduler_dict = {
        'step_lr': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        'one_cycle_lr': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=step_size, epochs=40),
        'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=min_lr),
        'multi_step_lr': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[step_size, step_size*2], gamma=gamma),
        'cyclic_lr': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=0.1, step_size_up=step_size, cycle_momentum=False),
    }

    scheduler = scheduler_dict.get(scheduler_name.lower())

    return scheduler

class EarlyStopping: 
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=7, min_delta=0.001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_metric = None
        self.counter = 0

    def __call__(self, val_loss):
        current_metric = val_loss 

        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if self.monitor == 'val_loss' and current_metric < (self.best_metric - self.min_delta):
            self.best_metric = current_metric
            self.counter = 0
            print(f"Validation loss decreased to {current_metric:.5f}, resetting patience")

        else:
            self.counter += 1
            print(f"No improvement in {self.monitor}, patience {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            print("Early stopping reached max patience")
            return True

        return False