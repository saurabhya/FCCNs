import torch
import networks as models
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tf
from utils import *


def accuracy(output, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
