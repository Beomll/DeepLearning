import torch
import torch.nn as nn

x_data = torch.tensor([[0.1, 2.3]], dtype=torch.float32)
linear = nn.Linear(2, 1)

print("linear.weight: ", linear.weight)
print("linear.bias: ", linear.bias)

print("torch.matmul(x_data, linear.weight.T): ", torch.matmul(x_data, linear.weight.T))
print("torch.matmul(x_data, linear.weight.T) + linear.bias: ", torch.matmul(x_data, linear.weight.T) + linear.bias)

hypothesis = torch.sign(linear(x_data))
print("hypothesis: ", hypothesis)

print(torch.sign(torch.matmul(x_data, linear.weight.T) + linear.bias))