import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from tqdm import tqdm

from parameter_server_pb2 import Gradient, Weight, Model
from proto_utils import load_proto, model_to_proto

def compare_models(model1, model2):
    cpu = torch.device('cpu')
    model1, model2 = model1.to(cpu), model2.to(cpu)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def create_model():
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 10)
    return model


def train_model(model, criterion, optimizer, train_loader):
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)

    # Training for 1 epoch
    cum_loss = 0.0
    correct = 0
    model.train()

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, pred = outputs.max(1)
            correct += (pred == y).sum().item()
            cum_loss += loss.item()


    n_train = len(train_loader.dataset)
    print(f"Finished in {time.time() - start_time} seconds.")
    print(f"Train acc={correct / n_train}, train loss={cum_loss / n_train}.")


### DATA 

class ToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1)
        # angle = random.choice(self.angles)
        # return TF.rotate(x, angle)

preprocessFn = transforms.Compose(
    [transforms.Resize(256), 
     transforms.CenterCrop(224), 
     transforms.ToTensor(), 
     ToRGB(),
     transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                          std = [0.229, 0.224, 0.225])])


mnist_train = datasets.MNIST('./data', train=True,
    transform=preprocessFn, download=True)
mnist_test = datasets.MNIST('./data', train=False,
    transform=preprocessFn, download=True)

batch_size = 128 
train_loader = DataLoader(mnist_train, batch_size = batch_size, 
                         shuffle = True, num_workers = 0)
val_loader = DataLoader(mnist_test, batch_size = batch_size, 
                       shuffle = False, num_workers = 0)


### MODEL DEFINITION
model = create_model()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(model)
model_proto = model_to_proto(model, log=True)
# print(len(model_proto.weights))
# print(pickle.loads(model_proto.weights[0].value))

print([name for name, _ in model.named_parameters()])

# Train for 1 epoch and save to proto
train_model(model, criterion, optimizer, train_loader)
model_proto = model_to_proto(model)


# Create new model and load from protobuf
model2 = create_model()
optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
load_proto(model2, model_proto)
print("Models are the same?", compare_models(model, model2))
train_model(model2, criterion, optimizer2, train_loader)

# Reset to 1 epoch completed
load_proto(model2, model_proto)
train_model(model2, criterion, optimizer2, train_loader)

# Train for another epoch
train_model(model2, criterion, optimizer2, train_loader)