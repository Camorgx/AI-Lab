from train import *

import torch

model = torch.load('../output/model.pth')
while True:
    inputs = input()
    if len(inputs) == 0:
        break
    generate(model, inputs)
