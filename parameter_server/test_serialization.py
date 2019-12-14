import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle

from parameter_server_pb2 import Gradient, Weight, Model



model = models.resnet18()

params = list(model.parameters())
print(list(model.named_parameters())[:2])

print(len(params))
print(params[0].shape)
# for param in model.parameters():
print("====Gradients")
print(params[0].grad)



x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x*x + 2

print(y)
print(y.grad_fn)

out = y.mean()

print(out.grad_fn)

out.backward()
print(x.grad)
print(y.grad)

print(pickle.dumps(x.grad))
print(pickle.loads(pickle.dumps(x.grad)))



start = time.time()
model_proto = Model()
for idx, param in enumerate(model.parameters()):
    weight = model_proto.weights.add()
    weight.index = idx
    weight.value = pickle.dumps(param)
print(f"Model serialized in {time.time() - start} seconds")

start = time.time()
with torch.no_grad():
    for i, weight in enumerate(model.parameters()):
        loaded_tensor = pickle.loads(model_proto.weights[i].value).data
        weight.copy_(loaded_tensor)
print(f"Model deserialized in {time.time() - start} seconds")



# print(dir(pickle.loads(model_proto.weights[0].value)))