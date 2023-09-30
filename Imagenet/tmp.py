import torch
from utils import *
import networks as models

model = models.resnet152(num_classes= 1000)
count_parameters(model)


