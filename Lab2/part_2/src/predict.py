from train import *

import torch

model = torch.load('../output/model_10000.pth')
while True:
    inputs = input()
    if len(inputs) == 0:
        break
    generate(model, inputs)
