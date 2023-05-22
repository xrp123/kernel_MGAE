# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python3.6

from graph import CreateAdjacency, CreateDegree
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import geopandas as gp


def feature_extract(shpfp, csvfp):
    """
    由于使用arcgis直接导出的csv中存在中文乱码的现象
    因此使用本函数进行csv文件utf-8格式的导出
    :param shpfp:
    :param csvfp:
    :return:
    """
    data = gp.read_file(shpfp)
    data = data.loc[:, 'Science':'Culture']
    data.to_csv(r"{}/dltb_origin.csv".format(csvfp),
                index=None,
                encoding='utf-8-sig')


def feature_standard(OriginData):
    """
    对属性数据进行初始化
    将其划为[0,1]
    :param OriginData:
    :return:
    """
    min = np.min(OriginData)
    max = np.max(OriginData)
    OutData = (OriginData - min) / (max - min)
    return OutData


def LabelTocsv(y_pred, filepath):
    """
    输出标签文件
    文件格式为csv
    :param y_pred: 标签数据
    :param filepath: 导出地址
    :return:
    """
    label = pd.DataFrame({'label': y_pred})
    label.reset_index(inplace=True)
    label.to_csv(filepath,
                 index=False,
                 encoding='utf_8_sig')


def BuildMatrix(Comsidefile):
    """
    权重连接矩阵
    :param Comsidefile:
    :return:
    """
    matrix = CreateAdjacency(Comsidefile)

    # 自连接
    left = range(matrix.shape[0])
    right = range(matrix.shape[0])
    I = np.ones(matrix.shape[0])
    I = coo_matrix((I, (left, right)), shape=(matrix.shape[0], matrix.shape[0]))
    matrix = matrix + I

    degree = CreateDegree(matrix)

    degree = degree.diagonal()
    degree = degree ** (-0.5)
    degree = coo_matrix((degree, (left, right)), shape=(degree.shape[0], degree.shape[0]))

    W = (degree.dot(matrix)).dot(degree)

    return W
