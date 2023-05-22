# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python3.6

import torch
import numpy as np
from graph import sparse_mx_to_torch_sparse_tensor


def mDA(preX, noise, e, W):
    torch.cuda.empty_cache()
    n = preX.shape[0]
    d = preX.shape[1]
    preXT = preX.T
    preXTb = np.insert(preXT, d, np.ones(shape=n), axis=0)
    preXT = torch.from_numpy(preXT).cuda()
    preXTb = torch.from_numpy(preXTb).cuda()
    W = sparse_mx_to_torch_sparse_tensor(W).cuda()

    S = torch.matmul(preXT, preXT.t())
    Sp = torch.matmul(torch.mm(W, preXTb.t()).t(), preXTb.t())
    temp1 = torch.matmul(W, preXTb.t()).t()
    temp2 = torch.matmul(W.t(), temp1.t()).t()
    Sq = torch.matmul(temp2, preXTb.t())

    q = np.ones(shape=d + 1)
    q = q * (1 - noise)
    q[d] = 1
    q = np.mat(q).T
    q = torch.from_numpy(q).cuda()

    Q = torch.mul(Sq, torch.mm(q, q.t()))

    diagSq = torch.diagonal(Sq)
    diagSq = torch.unsqueeze(diagSq, 1)
    temp = torch.mul(q, diagSq).t()
    rowlength = Q.shape[0]
    collength = Q.shape[1]
    row = list(range(rowlength))
    col = list(range(collength))
    Q[row, col] = temp

    temp = q.t().repeat(d, 1)
    P = torch.mul(Sp[0:d, :], temp)

    reg = torch.eye(d + 1).cuda() * e
    reg[d, d] = 0
    Q = Q + reg
    Q = torch.inverse(Q)
    Weight = torch.mm(P, Q)

    newX = torch.mm(W.t(), torch.mm(Weight, preXTb).t()).t()

    return newX.t()
