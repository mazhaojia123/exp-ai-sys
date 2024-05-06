import torch
import numpy as np

def func(path):
    weight=torch.load(path, map_location='cpu')
    for k,v in weight.items():
        print(k, v.size())
        np.savetxt(k, v.numpy().flatten())

if __name__=="__main__":
    func('./pytorch_model.bin')
