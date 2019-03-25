import numpy as np
import torch
from tensor_method import contra
import time

d = 3000
r = 10
learning_rate = 0.1
loops = 30

A = torch.rand(d, d)
E = torch.rand(d, d)
M = torch.rand(d, r)
N = torch.rand(r, d)

A = A/A.norm()
E = E/E.norm()
M = M/M.norm()
N = N/N.norm()

M.requires_grad = True
N.requires_grad = True

t1 = time.time()
for loop in range(loops):

    A_appro = M @ N
    L = (torch.trace(A@E) - torch.trace(A_appro@E))**2
    L.backward()
    
    with torch.no_grad():
        M -= learning_rate * M.grad
        N -= learning_rate * N.grad
        M.grad.zero_()
        N.grad.zero_()

t2 = time.time()
print(t2-t1)
