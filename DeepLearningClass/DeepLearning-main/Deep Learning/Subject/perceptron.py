# step function, sigmoid, ReLU, 입력값 3차원으로

import torch
import numpy as np
import torch.nn as nn

ReLU = nn.ReLU()
Sigmoid = nn.Sigmoid()

x_data = torch.tensor([[0.1, 2.3, 3.5]], dtype=torch.float32)
linear = nn.Linear(3, 1)

# Step Function
hypothesis_step = ReLU(torch.sign(linear(x_data)))

# ReLU
hypothesis_ReLU = ReLU(linear(x_data))

# Sigmoid
hypothesis_sigmoid = torch.sigmoid(linear(x_data))

print("hypothesis_StepFunction: ", hypothesis_step)
print("hypothesis_ReLU: ", hypothesis_ReLU)
print("hypothesis_sigmoid: ", hypothesis_sigmoid)