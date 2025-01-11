import torch
import json
from torchvision import transforms, models
from PIL import Image

import numpy as np

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
    model = torch.jit.load("model.pth")
    model.eval()

    return model

# species_dict = pickle.load(open("species_dict_new.pkl", "rb"))
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


def predict_species(fileName, k=5):

    # Load the image
    image = Image.open(fileName)

    # Get the size of the image
    width, height = image.size

    image = preprocess(image)
    image = image.unsqueeze(0)  # Add an extra dimension for batch size

    # Perform the inference
    with torch.no_grad():
        output = species_model(image)

    # Interpret the output (assumes the model outputs raw scores, not probabilities)
    _, predicted = torch.max(output, 1)

    species_dict = ['robin', 'sparrow', 'tit']
    pred_class = species_dict[predicted.item()]

    # Convert output scores to probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    return probabilities, pred_class


if __name__ == "__main__":

    # Some quick tests

    for fileName in [
        "/home/dan/birds2024/tit/motion_frame_20240523_144456_0_BORNEAN_BRISTLEHEAD.png",
        "/home/dan/birds2024/tit/motion_frame_20240206_090341_0_HIMALAYAN_BLUETAIL.png",
        "/home/dan/birds2024/tit/motion_frame_20241103_111630_0_OSTRICH.png",
        "/home/dan/birds2024/tit/motion_frame_20241103_111608_0_SCARLET_FACED_LIOCICHLA.png",
        "/home/dan/birds2024/tit/motion_frame_20241103_151105_0_SCARLET_FACED_LIOCICHLA.png",
        "/home/dan/birds2024/tit/motion_frame_20240521_190717_0_HIMALAYAN_BLUETAIL.png",
        "/home/dan/birds2024/tit/motion_frame_20240522_114615_0_SCARLET_FACED_LIOCICHLA.png",
        "/home/dan/birds2024/tit/motion_frame_20240519_191410_0_OYSTER_CATCHER.png",
        "/home/dan/birds2024/sparrow/motion_frame_20240526_055514_0_HIMALAYAN_BLUETAIL.png",
        "/home/dan/birds2024/robin/motion_frame_20241103_064537_0_SCARLET_FACED_LIOCICHLA.png",
        "/home/dan/birds2024/sparrow/motion_frame_20240626_120618_0_GILDED_FLICKER.png",
        "/home/dan/data/birds/test/tit/motion_frame_20230809_094755_3.png",
        "/home/dan/data/birds/test/tit/motion_frame_20230809_133046_2.png",
        "/home/dan/data/birds/test/tit/motion_frame_20230809_133029_0.png",
        "/home/dan/motion_frame_20241231_152736_0_SCARLET_FACED_LIOCICHLA.png",
        '/home/dan/motion_frame_20241231_150746_0_SCARLET_FACED_LIOCICHLA.png',
        "/home/dan/motion_frame_20241231_152625_0_SCARLET_FACED_LIOCICHLA.png",
    ]:

        p, c = predict_species(fileName, 2)

        print(p, c)
