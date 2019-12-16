from concurrent import futures
import time
import math
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

from proto_utils import model_to_proto, update_gradients
from torch_utils import create_model
import parameter_server_pb2
import parameter_server_pb2_grpc


class ParameterServerServicer(parameter_server_pb2_grpc.ParameterServerServicer):
    """Provides methods that implement functionality of the parameter server."""

    def __init__(self):
        self.model = create_model()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def UpdateGradients(self, request, context):
        update_gradients(self.model, request)
        optimizer.step()
        return parameter_server_pb2.GradientUpdateResponse()

    def GetModel(self, request, context):
        return model_to_proto(self.model)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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