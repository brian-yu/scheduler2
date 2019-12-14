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
from proto_utils import load_proto, model_to_proto


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


proto = model_to_proto(model)
print(len(proto.weights))


load_proto(model, proto)



# print(dir(pickle.loads(model_proto.weights[0].value)))