import logging
import time

import grpc

from proto_utils import load_proto, model_to_proto, gradients_to_proto
from torch_utils import create_model
import parameter_server_pb2
import parameter_server_pb2_grpc

# options = [('grpc.max_message_length', 100 * 1024 * 1024)]
options = [('grpc.max_receive_message_length', 500 * 2**20)]
channel = grpc.insecure_channel('localhost:50051', options=options)
stub = parameter_server_pb2_grpc.ParameterServerStub(channel)

start = time.time()
model_proto = stub.GetModel(parameter_server_pb2.ModelRequest())
print(f"Model retrieved in {time.time() - start} s")

model = create_model()
load_proto(model, model_proto)

print(model)