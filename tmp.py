import torch
import networks as models

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = models.CDS_E(num_classes= 10)

print(f"Total number of model parameters: {count_params(model)}")

