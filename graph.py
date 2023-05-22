# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python3.6

import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import torch
import time


def CreateAdjacency(filepath):
    """
    通过公共边构建邻接矩阵
    :param filepath:公共边文件 -> .csv
    :return:邻接矩阵 -> csr
    """
    start = time.time()
    df = pd.read_csv(filepath)
    index = df['FID']
    left = df['LEFT_FID']
    right = df['RIGHT_FID']

    # 获取多边形上下限
    left_max = left.max()
    left_min = left.min()
    right_max = right.max()
    right_min = right.min()
    list_maxmin = [left_max, left_min, right_min, right_max]
    max = int(np.max(list_maxmin))
    min = int(np.min(list_maxmin))

    # 构建稀疏矩阵
    row = np.array(left)
    col = np.array(right)
    data = np.ones(len(index))
    coo = coo_matrix((data, (row, col)), shape=(max + 1, max + 1))
    csr = coo.tocsr()  # 使用csr稀疏矩阵进行运算
    csr[csr > 1] = 1  # 去除重复的公共边
    csr = csr + csr.T.multiply(csr.T > csr) - csr.multiply(csr.T > csr)  # 将稀疏矩阵转为对称矩阵，即无向图
    print("Build Graph, time:{:.2f}s".format(time.time() - start))

    return csr


def CreateDegree(csr):
    """
    计算矩阵的度矩阵
    :param csr: csr格式的邻接矩阵
    :return: coo格式的度矩阵
    """
    D_data = csr.sum(1).flatten().A
    D_data = np.squeeze(D_data)
    D_data = D_data.astype(np.int32)
    min = 0
    max = csr.shape[0]
    row = np.linspace(min, max - 1, max)
    row = row.astype(np.int32)
    col = np.linspace(min, max - 1, max)
    col = col.astype(np.int32)
    D = coo_matrix((D_data, (row, col)), shape=(max, max))
    return D


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为torch稀疏张量
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float64)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
