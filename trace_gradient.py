import numpy as np
import torch
from tensor_method import contra
import time

d1=3000
d2=3000
r=10
learning_rate=0.1
loops=30
his_loss=torch.zeros(1,loops);
A=torch.rand(d1,d2)
E=torch.rand(d1,d2)
M=torch.rand(d1,r)
N=torch.rand(r,d2)
A=A/A.norm()
E=E/E.norm()
M=M/M.norm()
N=N/N.norm()
t1=time.time()

for loop in range(loops):
    A0 = (contra(M, [1, 2], N, [2, 3]))[0]
    loss = (contra((A - A0), [1, 2], E, [1, 2]))[0]
    grad_M = 2 * loss * contra(E, [0, 2], N, [1, 2])[0]
    grad_N = 2 * loss * contra(M, [1, 2], E, [1, 3])[0]
    M = M + learning_rate * grad_M
    N = N + learning_rate * grad_N
    his_loss[0, loop] = loss ** 2;
    loss1 = (contra((A - A0), [1, 2], E, [1, 2]))[0]
    # loss2 = np.sqrt((float((torch.trace(A @ E) - torch.trace(M @ N @ E)))**2))
    # print(loss1 / loss2)
    

t2=time.time()
print(his_loss, t2-t1)
