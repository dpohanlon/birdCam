import torch
import json
from torchvision import transforms
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

bird_indices = mobilenetBirds().values()

birdModel = models.mobilenet_v3_large(pretrained=True)
birdModel.eval()

def mobilenetBirds():

    allCats = json.load(open('simple_labels.json', 'rb'))
    onlyBirds = json.load(open('birds.json', 'rb'))

    catIndices = {c : i for i, c in enumerate(allCats)}
    birdIndices = {c : catIndices[c] for c in onlyBirds}

    return birdIndices

def predict_bird(image_path):

    # Open image and apply preprocessing
    img = Image.open(image_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Predict
    with torch.no_grad():
        out = birdModel(batch_t)

    # Get top 5 indices
    _, indices = torch.topk(out, 5)

    # Check if any of the top 5 predictions is a bird
    is_bird = any(idx in bird_indices for idx in indices[0])

    return is_bird


def classifySpecies(fileName):

    # Load the quantized model
    model = torch.jit.load("quantized_model.pth")
    model.eval()

    class_dict = pickle.load(open("class_dict.pkl", "rb"))

    # Load the image
    image = Image.open(fileName)  # Assuming your image is named 'input.jpg'

    # Get the size of the image
    width, height = image.size

    # Crop 150px off the bottom
    image = image.crop((0, 0, width, height - 150))

    image = preprocess(image)
    image = image.unsqueeze(0)  # Add an extra dimension for batch size

    # Perform the inference
    with torch.no_grad():
        output = model(image)

    # Interpret the output (assumes the model outputs raw scores, not probabilities)
    _, predicted = torch.max(output, 1)
    print(f"Predicted class: {class_dict[predicted.item()]}")
    print("")

    # Convert output scores to probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the indices of the top 5 predictions
    top5_pred = torch.topk(probabilities, 5)

    # Print the top 5 classes and their probabilities
    for i in range(5):
        print(
            f"Class: {class_dict[top5_pred.indices[0][i].item()]}, Probability: {top5_pred.values[0][i]}"
        )

    return probabilities


if __name__ == "__main__":

    for fileName in [
        "t1.png",
        "t2.png",
        "t3.png",
        "t4.png",
        "t5.png",
        "t6.png",
        "t7.png",
        "t8.png",
    ]:

        classifySpecies(fileName)
