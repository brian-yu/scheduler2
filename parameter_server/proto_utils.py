import time
import pickle
import torch

from parameter_server_pb2 import Gradient, GradientUpdate, Weight, Model

def model_to_proto(model, log=False):
    start = time.time()
    model_proto = Model()
    for idx, param in enumerate(model.parameters()):
        weight = model_proto.weights.add()
        weight.index = idx
        weight.value = pickle.dumps(param)
    if log:
        print(f"Model serialized in {time.time() - start} seconds")
    return model_proto

def load_proto(model, proto, log=False):
    start = time.time()
    with torch.no_grad():
        for i, weight in enumerate(model.parameters()):
            loaded_tensor = pickle.loads(proto.weights[i].value).data
            weight.copy_(loaded_tensor)
    # for i, weight in enumerate(model.parameters()):
    #     weight.data = pickle.loads(proto.weights[i].value).data
    if log:
        print(f"Model deserialized in {time.time() - start} seconds")



def gradients_to_proto(model, log=False):
    start = time.time()
    gradient_proto = GradientUpdate()
    for idx, param in enumerate(model.parameters()):
        gradient = gradient_proto.gradients.add()
        gradient.index = idx
        gradient.value = pickle.dumps(param.grad)
    if log:
        print(f"Model gradients serialized in {time.time() - start} seconds")
    return gradient_proto

def update_gradients(model, log=False):
    start = time.time()
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            gradient = pickle.loads(proto.gradients[i].value)
            param.grad = gradient
    if log:
        print(f"Gradients updated in {time.time() - start} seconds")