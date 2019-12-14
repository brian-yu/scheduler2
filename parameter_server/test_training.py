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

from parameter_server_pb2 import Gradient, Weight, Model

def model_to_proto(model):
    start = time.time()
    model_proto = Model()
    for idx, param in enumerate(model.parameters()):
        weight = model_proto.weights.add()
        weight.index = idx
        weight.value = pickle.dumps(param)
    print(f"Model serialized in {time.time() - start} seconds")
    return model_proto

def load_proto(model, proto):
    start = time.time()
    with torch.no_grad():
        for i, weight in enumerate(model.parameters()):
            loaded_tensor = pickle.loads(proto.weights[i].value).data
            weight.copy_(loaded_tensor)
    print(f"Model deserialized in {time.time() - start} seconds")

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
    model.to(device)


    # Training for 1 epoch
    cum_loss = 0.0
    correct = 0
    model.train()
    i = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        print(f"{i} / {len(train_loader)}")
        i += 1
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, pred = outputs.max(1)
            correct += (pred == y).sum().item()
            cum_loss += loss.item()




    # for i, data in enumerate(train_loader):
    #     inputs, labels = data[0].to(device), data[1].to(device)

    #     optimizer.zero_grad()

    #     outputs = model(inputs)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
        
    #     with torch.no_grad():
    #       _, max_labels = outputs.max(1)
    #       correct += (max_labels == labels).sum().item()
    #       cum_loss += loss.item()

    n_train = len(train_loader.dataset)
    print(f"Finished in {time.time() - start_time} seconds.")
    print(f"Train acc={correct / n_train}, train loss={cum_loss / n_train}.")
    # train_acc.append(correct / n_train)
    # train_loss.append(cum_loss / n_train)


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


train_model(model, criterion, optimizer, train_loader)

model_proto = model_to_proto(model)



model2 = create_model()
load_proto(model2, model_proto)


print("Models are the same?", compare_models(model, model2))


train_model(model2, criterion, optimizer, DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 0))


# load_proto(model, model_proto)
# train_model(model, criterion, optimizer, DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 0))

