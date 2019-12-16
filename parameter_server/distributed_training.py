import time
import os
import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import grpc

from parameter_server_pb2 import Gradient, Weight, Model, ModelRequest
from proto_utils import load_proto, model_to_proto, gradients_to_proto
from torch_utils import compare_models, create_model
import parameter_server_pb2_grpc


def train_model(model, criterion, train_loader, ps_stub):
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
        # optimizer.zero_grad()

        print("Fetching model")
        model_proto = ps_stub.GetModel(ModelRequest())
        load_proto(model, model_proto)
        print("Model fetched")

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        # optimizer.step()

        print("Sending gradients")
        ps_stub.UpdateGradients(gradients_to_proto(model))
        print("Gradients sent")

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
criterion = nn.CrossEntropyLoss()

options = [
    ('grpc.max_receive_message_length', 500 * 2**20),  # 500 MB limit
    ('grpc.max_message_length', 500 * 2**20)  # 500 MB limit
    ]
channel = grpc.insecure_channel('localhost:50051', options=options)
ps_stub = parameter_server_pb2_grpc.ParameterServerStub(channel)

train_model(model, criterion, train_loader, ps_stub)
