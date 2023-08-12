import torch
import json
from torchvision import transforms, models
from PIL import Image

import pickle

# Define the transformation
preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize the short side to 256
        transforms.CenterCrop(224),  # Crop a 224x224 square from the center
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize the tensor
    ]
)


def mobilenetBirds():

    allCats = json.load(open("simple_labels.json", "rb"))
    onlyBirds = json.load(open("birds.json", "rb"))

    catIndices = {c: i for i, c in enumerate(allCats)}
    birdIndices = {c: catIndices[c] for c in onlyBirds}

    return birdIndices

def speciesModel():

    # Load the quantized model
    model = torch.jit.load("quantized_model.pth")
    model.eval()

    return model

species_dict = pickle.load(open("class_dict.pkl", "rb"))
species_model = speciesModel()

bird_indices = mobilenetBirds().values()

birdModel = models.mobilenet_v3_large(pretrained=True)
birdModel.eval()


def predict_bird(image_path):

    # Open image and apply preprocessing
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Predict
    with torch.no_grad():
        out = birdModel(batch_t)

    probs = torch.nn.functional.softmax(out, dim=1).ravel()

    # Get top 5 indices
    probs5, indices5 = torch.topk(probs, 5)

    # Check if any of the top 5 predictions is a bird
    is_bird = any(idx in bird_indices for idx in indices5)

    birdProbs = probs[list(bird_indices)]

    return is_bird, birdProbs


def predict_species(fileName):

    # Load the image
    image = Image.open(fileName)

    # Get the size of the image
    width, height = image.size

    # Crop 150px off the bottom
    image = image.crop((0, 0, width, height - 150))

    image = preprocess(image)
    image = image.unsqueeze(0)  # Add an extra dimension for batch size

    # Perform the inference
    with torch.no_grad():
        output = species_model(image)

    # Interpret the output (assumes the model outputs raw scores, not probabilities)
    _, predicted = torch.max(output, 1)
    pred_class = species_dict[predicted.item()]

    # Convert output scores to probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the indices of the top 5 predictions
    top5_pred = torch.topk(probabilities, 5)

    return probabilities, pred_class


if __name__ == "__main__":

    # for fileName in [
    #     "t1.png",
    #     "t2.png",
    #     "t3.png",
    #     "t4.png",
    #     "t5.png",
    #     "t6.png",
    #     "t7.png",
    #     "t8.png",
    # ]:
    #
    #     classifySpecies(fileName)

    import os

    files = os.listdir("/Users/dan/Downloads/webcam_images/motion_images")
    files = list(filter(lambda x: ".png" in x, files))

    from tqdm import tqdm
    import numpy as np

    for f in tqdm(files):
        fileName = f"/Users/dan/Downloads/webcam_images/motion_images/{f}"
        is_bird, probs = predict_bird(fileName)
        if is_bird and np.sum(probs.ravel().numpy()) > 0.25:
            print(fileName)
            # print(probs)
            print("")
