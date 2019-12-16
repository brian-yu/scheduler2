import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


def compare_models(model1, model2):
    cpu = torch.device('cpu')
    model1, model2 = model1.to(cpu), model2.to(cpu)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def create_model(arch="alexnet"):
    if arch == "resnet":
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        return model
    elif arch == "alexnet":
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, 10)
        return model
    else:
        raise Exception("Invalid model architecture.")