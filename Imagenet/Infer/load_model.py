import torch
import networks as models
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tf
from utils import *


model = models.resnet18(num_classes=1000)

image = Image.open("image.png").convert("RGB")


transforms = tf.Compose(
    [
        tf.Resize((224, 224)),
        tf.ToTensor(),
        ToHSV(),
        ToComplex(),
    ]
)

img_tensor = transforms(image).unsqueeze(0)

# load model
filename = "model_best.pth.tar"
device = "cpu"
checkpoint = torch.load(filename, map_location=device)

# load model weights
model.load_state_dict(checkpoint["state_dict"])

output = model(img_tensor)


# check which class using argmax
print(output.abs().argmax(1))
