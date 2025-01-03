from pprint import pprint

import re

import os

import matplotlib as mpl
from torchvision.transforms.transforms import RandomApply

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from tqdm import tqdm

import time

import copy

import numpy as np

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

from collections import Counter

import pandas as pd

import torch
import torchvision

from torchvision.models import (
    mobilenet_v3_small,
    # MobileNet_V3_Small_QuantizedWeights,
)
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision import transforms, datasets

from torch import nn

import torch.optim as optim

from torch.quantization import convert

## Update training metadata CSV to include new bird images
## Update species_dict/class_dict to add new birds (corresponding to class ids in CSV) for evaluation

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from itertools import cycle

from collections import Counter

def get_class_weights_from_dataset(dataset):
    # Get the class counts from the dataset
    class_counts = Counter(dataset.targets)
    total_samples = len(dataset)
    num_classes = len(dataset.classes)
    class_weights = []

    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count == 0:
            weight = 0  # Handle classes with zero samples if any
        else:
            weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights_tensor


def compute_roc(y_true, y_pred_probs, n_classes):
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def compute_roc_for_selected_classes(y_true, y_pred_probs, selected_classes):
    n_classes = y_pred_probs.shape[1]

    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

    # Compute ROC curve and ROC area for selected classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in selected_classes:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for selected classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(selected_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Selected Classes')
    plt.legend(loc="lower right")
    plt.savefig('roc.pdf')
    plt.clf()

def compute_pr_curve_for_selected_classes(y_true, y_pred_probs, selected_classes):
    n_classes = y_pred_probs.shape[1]

    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

    # Compute Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in selected_classes:
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])

    # Plot the Precision-Recall curves for selected classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(selected_classes, colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0} (avg. precision = {1:0.2f})'.format(i, average_precision[i]))

    # Calculate the random chance precision
    random_precision = y_true_bin.sum(axis=0) / y_true_bin.shape[0]
    for i, color in zip(selected_classes, colors):
        plt.hlines(random_precision[i], 0, 1, colors=color, linestyles='dashed',
                   label='Random Precision for class {0} ({1:0.2f})'.format(i, random_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for Selected Classes')
    plt.legend(loc="lower left")
    plt.savefig('pr.pdf')
    plt.clf()


def compute_f1(y_true, y_preds):
    # Compute F1 score. Set average='macro' for multi-class tasks
    f1 = f1_score(y_true, y_preds, average='macro')
    print(f'F1 Score: {f1:.4f}')
    return f1

def imshow(inp, title=None, ax=None, figsize=(5, 5)):

    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


def visualize_model(model, rows=3, cols=3):
    was_training = model.training
    model.eval()
    current_row = current_col = 0
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    with torch.no_grad():
        for idx, (imgs, lbls) in enumerate(dataloaders["test"]):
            imgs = imgs.cpu()
            lbls = lbls.cpu()

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for jdx in range(imgs.size()[0]):
                imshow(imgs.data[jdx], ax=ax[current_row, current_col])
                ax[current_row, current_col].axis("off")
                ax[current_row, current_col].set_title(
                    "predicted: {}".format(class_names[preds[jdx]])
                )

                current_col += 1
                if current_col >= cols:
                    current_row += 1
                    current_col = 0
                if current_row >= rows:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def create_combined_model_mobilenetv3(model_fe, n_classes):
    # Step 1. Isolate the feature extractor.
    model_fe_features = nn.Sequential(
        # model_fe.quant,  # Quantize the input
        model_fe.features,  # Feature extraction layers
        model_fe.avgpool,
        # model_fe.dequant,  # Dequantize the output
    )

    # Step 2. Create a new "head"
    # num_ftrs = 960  # model_fe.classifier[3].in_features
    num_ftrs = 576  # model_fe.classifier[3].in_features

    new_head = nn.Sequential(
        # nn.Dropout(p=0.5),  # Increased dropout for regularization
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, 8),  # Intermediate layer
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Additional Dropout layer for regularization
        nn.Linear(8, n_classes)  # Final output layer
    )

    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        # model_fe_features,
        model_fe.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        new_head,
    )

    return new_model

    transforms.RandomApply

weird_transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.5, interpolation=3),
    transforms.RandomAdjustSharpness(0.9),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),]


data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),

        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),

        # These take a while so remove them when we have a lot of data
        # transforms.RandomApply(weird_transforms, 0.75),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device="cpu"):
    """
    Support function for model training.

    https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = []
    losses_test = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                losses.append(running_loss)
            if phase == "test":
                losses_test.append(running_loss)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best test Acc: {:4f}".format(best_acc))

    plt.plot(np.log(losses))
    plt.plot(np.log(losses_test))
    plt.savefig('loss.pdf')
    plt.clf()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Device', device)

data_dir = "/home/dan/data/birds2024"

metadata = pd.read_csv("/home/dan/birdCam/birdCam/classifier/bird_data.csv")

n_classes = len(metadata["class id"].unique())

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "test"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=1024, shuffle=True, num_workers=6, pin_memory=False
    )
    for x in ["train", "test"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets["train"].classes

print('Classes', class_names)
print('Dataset sizes', dataset_sizes)

# Calculate class sizes directly from the targets attribute
class_sizes = {}
for phase in ["train", "test"]:
    # Get all class labels in the dataset using the targets attribute
    class_counts = Counter(image_datasets[phase].targets)
    # Map these counts to class names
    class_sizes[phase] = {class_names[idx]: count for idx, count in class_counts.items()}

##

from collections import defaultdict
import random
from torch.utils.data import Subset

# Assuming image_datasets['train'] is your training dataset

# Step 1: Get class counts
class_counts = Counter(image_datasets['train'].targets)
min_class_count = min(class_counts.values())

# Step 2: Get indices of samples for each class
class_indices = defaultdict(list)
for idx, target in enumerate(image_datasets['train'].targets):
    class_indices[target].append(idx)

# Step 3: Subsample each class to the minimum class count
balanced_indices = []
for class_id, indices in class_indices.items():
    if len(indices) > min_class_count:
        balanced_class_indices = random.sample(indices, min_class_count)
    else:
        balanced_class_indices = indices
    balanced_indices.extend(balanced_class_indices)

# Step 4: Create a Subset of the dataset with these indices
balanced_train_dataset = Subset(image_datasets['train'], balanced_indices)

# Step 5: Create a DataLoader with this balanced dataset
balanced_train_loader = torch.utils.data.DataLoader(
    balanced_train_dataset, batch_size=1024, shuffle=True, num_workers=6, pin_memory=False
)

# Step 6: Update the dataloaders and dataset sizes
dataloaders['train'] = balanced_train_loader
dataset_sizes['train'] = len(balanced_train_dataset)

##

# Print the class sizes
for phase in ["train", "test"]:
    print(f"\nClass sizes in {phase} dataset:")
    for class_name, count in class_sizes[phase].items():
        print(f"{class_name}: {count}")


weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(weights=weights, progress=True, quantize=False)

print(model.classifier[3].in_features)

model.train()

model_ft = create_combined_model_mobilenetv3(model, n_classes)

for param in model_ft.parameters():
    param.requires_grad = False

# for param in model_ft[-3].parameters():  # Fine-tune penultimate layer
#     param.requires_grad = True

# for param in model_ft[-2].parameters():  # Fine-tune penultimate layer
#     param.requires_grad = True

for param in model_ft[-1].parameters():  # Fine-tune final layer
    param.requires_grad = True

model_ft.to(device)  # We can fine-tune on GPU if available

class_weights = get_class_weights_from_dataset(image_datasets['train'])
print('weights', class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

# Note that we are training everything, so the learning rate is lower
# Notice the smaller learning rate
optimizer_ft = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_ft.parameters()),
    lr=0.005,
    # weight_decay=0.1  # L2 regularization with weight decay
    # momentum=0.9,
    # weight_decay=0.1,
)

# Decay LR by a factor of 0.3 every several epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.5)

model_ft_tuned = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device=device
)

model_ft_tuned.cpu()

torch.jit.save(torch.jit.script(model_ft_tuned), "model.pth")

plt.savefig("trained.png")
plt.clf()

device = 'cpu'

model_ft_tuned.to(device)

# Assuming `dataloaders["test"]` is your test dataloader
y_true = []
y_pred_probs = []
y_preds = []

model_ft_tuned.eval()
with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        outputs = model_ft_tuned(inputs)

        # Get the predicted probabilities for the ROC curve
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred_probs.extend(probs)
        y_preds.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_preds = np.array(y_preds)

# Binary classification for now

# Compute the ROC curve
compute_roc_for_selected_classes(y_true, y_pred_probs, [0, 1, 2])

compute_pr_curve_for_selected_classes(y_true, y_pred_probs, [0, 1, 2])

# Compute the F1 score
compute_f1(y_true, y_preds)
