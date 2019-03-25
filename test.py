import torch
import math
import numpy as np

# chain_length = 100
# bond_dim = 100
# rank = 50

# matrix_chain = []
# for i in range(chain_length):
#     A = torch.rand(bond_dim, bond_dim)
#     matrix_chain.append(A / A.norm())

# real_value = torch.eye(bond_dim)
# for i in range(chain_length):
#     real_value = real_value @ matrix_chain[i]

# value = float(torch.trace(real_value))

# for i in np.arange(0, chain_length, 2):
#     mat = matrix_chain[i] @ matrix_chain[i+1]
#     u, s, v = torch.svd(mat)
#     for j in range(rank, chain_length):
#         s[j] = 0
#     s = torch.diag(s)
#     matrix_chain[i] = u
#     matrix_chain[i+1] = s @ v.t()

# real_value = torch.eye(bond_dim)
# for i in range(chain_length):
#     real_value = real_value @ matrix_chain[i]
# value_1 = float(torch.trace(real_value))

# err = (value - value_1) / value
# print(value, value_1, err)
A = torch.ones(10,10)
B = torch.sum(A**2)
print(B)