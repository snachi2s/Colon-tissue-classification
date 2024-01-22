from datasets import load_dataset
import numpy as np
#import albumentations
#import evaluate
from transformers import ViTImageProcessor
import transformers
import torch
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
#from transformers import DefaultDataCollator
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report


dataset = load_dataset("imagefolder", data_dir="ISM/ism_project_2023/data")

#------------------
# split
#------------------
ds = dataset.shuffle(seed=42)
ds = dataset['train'].train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
train_ds = ds['train']
test_ds = ds['test']


#------------------
# id2label
#------------------
labels = train_ds.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

#-----------------------------
# albumentations transforms
#-----------------------------

# transform = albumentations.Compose([
#     albumentations.HorizontalFlip(p=0.5),
#     albumentations.RandomBrightnessContrast(p=0.2),
# ])

# def transforms(examples):
#     examples["pixel_values"] = [
#         transform(image=np.array(image))["image"] for image in examples["image"]
#     ]

#     return examples

#--------------------------
# torchvision transforms
#--------------------------
    
model_name = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = transforms.Normalize(mean=image_mean, std=image_std)

train_transform = transforms.Compose([
      transforms.Resize((processor.size["height"], processor.size["width"])),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=.3, hue=.1)], p=0.3),
      transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(5, 9))], p=0.3),
      transforms.ToTensor(),
      normalize
      #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
  ])

test_transform = transforms.Compose([
    transforms.Resize((processor.size["height"], processor.size["width"])),
    transforms.ToTensor(),
    normalize
])

def train_transforms(examples):
    examples['pixel_values'] = [train_transform(image.convert("RGB")) for image in examples['image']]
    return examples

def test_transforms(examples):
    examples['pixel_values'] = [test_transform(image.convert("RGB")) for image in examples['image']]
    return examples

train_ds.set_transform(train_transforms) #adds new key --> pixel_values(transforms applied)
test_ds.set_transform(test_transforms)

#print(train_ds[0]['pixel_values'].shape)

#-------------------------
# HF model preprocessing
#-------------------------

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }

#--------------------------
# metric definition
#--------------------------

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    print(classification_report(labels, predictions))
    return dict(accuracy=accuracy_score(predictions, labels))

#----------
# MODEL
#----------
checkpoint = model_name 
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

#--------------------------
# TRAINING
#--------------------------

training_args = TrainingArguments(
  output_dir="./vit-colon-cancer-classification-latest",
  per_device_train_batch_size=10,
  evaluation_strategy="steps",
  num_train_epochs=10,
  #fp16=True,
  save_steps=100,
  #eval_steps=100,
  logging_steps=100,
  learning_rate=3e-6,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to=['tensorboard'],
  load_best_model_at_end=True,
  metric_for_best_model='accuracy',
  greater_is_better=True,
  do_eval=True,
  gradient_accumulation_steps=1,
  lr_scheduler_type='linear',
  #warmup_ratio=0.1,
  weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=6)]
)

trainer.train()

outputs = trainer.predict(test_ds)
print(outputs.metrics)

trainer.push_to_hub()
