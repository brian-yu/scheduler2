import time
import pickle
import torch

from parameter_server_pb2 import Gradient, GradientUpdate, Weight, Model

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
    # for i, weight in enumerate(model.parameters()):
    #     weight.data = pickle.loads(proto.weights[i].value).data
    print(f"Model deserialized in {time.time() - start} seconds")



def gradients_to_proto(model):
    start = time.time()
    gradient_proto = GradientUpdate()
    for idx, param in enumerate(model.parameters()):
        gradient = gradient_proto.gradients.add()
        gradient.index = idx
        gradient.value = pickle.dumps(param.grad)
    print(f"Model gradients serialized in {time.time() - start} seconds")
    return gradient_proto