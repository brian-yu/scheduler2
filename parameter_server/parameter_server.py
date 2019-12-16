from concurrent import futures
import time
import logging

import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from collections import defaultdict
import numpy as np
import torchvision

from proto_utils import model_to_proto
from torch_utils import create_model
import parameter_server_pb2
import parameter_server_pb2_grpc

def update_gradients(model, proto, log=False):
    start = time.time()
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            gradient = pickle.loads(proto.gradients[i].value)
            param.grad = gradient
    if log:
        print(f"Gradients updated in {time.time() - start} seconds")


class ParameterServerServicer(parameter_server_pb2_grpc.ParameterServerServicer):
    """Provides methods that implement functionality of the parameter server."""

    def __init__(self):
        self.model = create_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.gradients = []
        self.threshold = 5

    def UpdateGradients(self, request, context):
        self.optimizer.zero_grad()
        update_gradients(self.model, request)
        self.optimizer.step()
        return parameter_server_pb2.GradientUpdateResponse()


        # self.gradients.append(gradient_update)
        # if len(self.gradients) >= self.threshold:
        #     print("Averaging gradients")
        #     self.average_gradients()

    def GetModel(self, request, context):
        return model_to_proto(self.model)


def serve():
    options = [('grpc.max_receive_message_length', 500 * 2**20)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(
        ParameterServerServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Parameter server starting...")
    server.start()
    print("Parameter server started.")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()