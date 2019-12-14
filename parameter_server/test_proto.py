from parameter_server_pb2 import Gradient, Weight, Model

gradient = Gradient()
gradient.index = 0
gradient.value = b'\x04\x00'

print(gradient)
print(gradient.index)
print(gradient.value)


weight = Weight()
weight.index = 0
weight.value = b'\x04\x00'

print(weight)
print(weight.index)
print(weight.value)


import random

model = Model()

for i in range(10000):
    weight = model.weights.add()
    weight.index = i
    weight.value = b'\x04\x00'

print(model)



