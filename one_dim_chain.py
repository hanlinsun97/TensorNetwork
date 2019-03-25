import numpy as np
import torch
import math
import copy

def contra_chain(chain_length, bond_dim, chain):
    real_value = torch.eye(bond_dim)
    for i in range(chain_length):
        real_value = real_value @ chain[i]
    value = float(torch.trace(real_value))
    return value, real_value


def direct_variational(bond_dim, chain, i, rank, loops, learning_rate):
    MM = torch.rand(bond_dim, rank)
    NN = torch.rand(rank, bond_dim)
    MM = MM / MM.norm()
    NN = NN / NN.norm()
    MM.requires_grad = True
    NN.requires_grad = True
    for loop in range(loops):
        # K = (torch.trace(Mat_2) - torch.trace(MM@NN))**2
        K = torch.sum((chain[i] - MM@NN)**2)
        K.backward(retain_graph=True)
        with torch.no_grad():
            MM -= learning_rate * MM.grad 
            NN -= learning_rate * NN.grad 
            MM.grad.zero_()
            NN.grad.zero_()
    return MM @ NN

def man_var(bond_dim, chain, i, rank, loops, learning_rate):
    M = torch.rand(bond_dim, rank)
    N = torch.rand(rank, bond_dim)
    M = M / M.norm()
    N = N / N.norm()
    for loop in range(loops):
        L = (torch.trace(chain[i]) - torch.trace(M@N))
        # print(L)
        grad_M = N.t() * L
        grad_N = M.t() * L
        M += learning_rate * grad_M * 2
        N += learning_rate * grad_N * 2
    return M @ N



def enviro_variational(bond_dim, rank, Mat_1, Mat_2, Mat_3, loops, learning_rate):
    M = torch.rand(bond_dim, rank)
    N = torch.rand(rank, bond_dim)
    M = M / M.norm()
    N = N / N.norm()
    M.requires_grad = True
    N.requires_grad = True
    for loop in range(loops):
        # L = (torch.trace(Mat_1@Mat_2@Mat_3) - torch.trace(Mat_1@M@N@Mat_3))**2
        L = torch.sum((Mat_1@Mat_2@Mat_3 - Mat_1@M@N@Mat_3)**2)
        L.backward(retain_graph=True)
        # print(float(L))
        with torch.no_grad():
            M -= learning_rate * M.grad
            N -= learning_rate * N.grad
            M.grad.zero_()
            N.grad.zero_()
    return M @ N

def svd_single(rank, bond_dim, chain, i):
    u, s, v = torch.svd(chain[i])
    for i in range(rank, bond_dim):
        s[i] = 0
    s = torch.diag(s)
    return u @ s @ v.t()

def svd_TRG(rank, bond_dim, chain, i):
    u, s, v = torch.svd(chain[i]@chain[i+1])
    for i in range(rank, bond_dim):
        s[i] = 0
    s = torch.diag(s)
    return u, s @ v.t()

def svd_SRG(rank, bond_dim, chain_length, chain, i):
    
    part_1 = torch.eye(bond_dim)
    part_2 = torch.eye(bond_dim)
    A = chain[i] @ chain[i+1]
    for j in range(i):
        part_1 = part_1 @ chain[j]
    for k in range(i+2, chain_length):
        part_2 = part_2 @ chain[k]
   
    E = part_2 @ part_1
  

    u_1, s_1, v_1 = torch.svd(E)
    s_1 = torch.diag(s_1)
    v_1 = v_1.t()
    M_hat = (s_1 ** (1/2)) @ v_1 @ A @ u_1 @ (s_1 ** (1/2))

    u_2, s_2, v_2 = torch.svd(M_hat)
    s_2 = torch.diag(s_2)
    v_2 = v_2.t()
    
    S_a = (s_2 ** (1/2)) @ u_2 @ (s_2 ** (-1/2)) @ v_2
    S_b = (s_2 ** (1/2)) @ v_2 @ (s_2 ** (-1/2)) @ u_2

    print(v_2)

    S_a = S_a[:, 0:rank]
    S_b = S_b[0:rank, :]
    low_A = S_a @ S_b
    print(low_A)
    u_3, s_3, v_3 = torch.svd(low_A)
    return u_3 @ np.diag(s_3), v_3.t()

def matrix_generator(chain_length, i, chain):
    if i == 0:
        Mat_1 = matrix_chain[chain_length-1]
        Mat_2 = matrix_chain[i]
        Mat_3 = matrix_chain[i+1]
    elif i == chain_length - 1:
        Mat_1 = matrix_chain[i-1]
        Mat_2 = matrix_chain[i]
        Mat_3 = matrix_chain[0]
    else:
        Mat_1 = matrix_chain[i-1]
        Mat_2 = matrix_chain[i]
        Mat_3 = matrix_chain[i+1]
    return Mat_1, Mat_2, Mat_3


chain_length = 100
bond_dim = 100
rank = 50
learning_rate = 0.1
loops = 100
matrix_chain = []
torch.manual_seed(1)

for i in range(chain_length):
    A = torch.rand(bond_dim, bond_dim)
    matrix_chain.append(A / A.norm())

chain_1 = copy.copy(matrix_chain)
chain_2 = copy.copy(matrix_chain)
chain_3 = copy.copy(matrix_chain)
chain_4 = copy.copy(matrix_chain)
chain_5 = copy.copy(matrix_chain)
chain_6 = copy.copy(matrix_chain)
chain_7 = copy.copy(matrix_chain)

value_1, _ = contra_chain(chain_length, bond_dim, chain_1) # Original

for i in range(chain_length):
    chain_2[i] = direct_variational(bond_dim, chain_2, i, rank, loops, learning_rate)
value_2, _ = contra_chain(chain_length, bond_dim,chain_2)   # single_variational

for i in range(chain_length):
    chain_3[i] = svd_single(rank, bond_dim, chain_3, i)
value_3, _ = contra_chain(chain_length, bond_dim, chain_3)   # single_svd

for i in range(chain_length):
    Mat_1, Mat_2, Mat_3 = matrix_generator(chain_length, i, chain_4)
    chain_4[i] = enviro_variational(bond_dim, rank, Mat_1, Mat_2, Mat_3, loops, learning_rate)
value_4, _ = contra_chain(chain_length, bond_dim, chain_4)   # Enviroment

for i in np.arange(0, chain_length, 2):
    chain_5[i], chain_5[i+1] = svd_TRG(rank, bond_dim, chain_5, i)
value_5, _ = contra_chain(chain_length, bond_dim, chain_5)

for i in np.arange(0, chain_length, 2):
    for j in np.arange(0, chain_length, 2):
        chain_re = copy.copy(chain_6)
        chain_re[j], chain_re[j+1] = svd_TRG(rank, bond_dim, chain_re, j)
    chain_6[i], chain_6[i+1]= svd_SRG(rank, bond_dim, chain_length, chain_re, i)
    # print(chain_6[i])
value_6, _ = contra_chain(chain_length, bond_dim, chain_6)

for i in range(chain_length):
    # Mat_1, Mat_2, Mat_3 = matrix_generator(chain_length, i, chain_7)
    chain_7[i] = man_var(bond_dim, chain_7, i, rank, loops, learning_rate)
value_7, _ = contra_chain(chain_length, bond_dim,chain_7)   # single_variational


print(value_1, value_2, value_3, value_4, value_5, value_7)
err_1 = np.abs(value_2 - value_1) / value_1
err_2 = np.abs(value_3 - value_1) / value_1
err_3 = np.abs(value_4 - value_1) / value_1
err_4 = np.abs(value_5 - value_1) / value_1
# err_5 = np.abs(value_6 - value_1) / value_1
err_6 = np.abs(value_7 - value_1) / value_1

print(err_1)
print(err_2)
print(err_3)
print(err_4)
# print(err_5)
print(err_6)
