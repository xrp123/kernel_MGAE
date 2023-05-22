# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python3.6

from func import feature_extract, feature_standard, BuildMatrix, LabelTocsv
import pandas as pd
from sklearn.cluster import KMeans
from MGAE import mDA


def main():
    # convert shp to csv
    dltbshpfp = r"data/feature/dltb.shp"
    dltbcsvfp = r"data/feature/dltb_origin.shp"
    feature_extract(dltbshpfp, dltbcsvfp)

    # standard
    dltborigin = pd.read_csv(dltbcsvfp)
    dltbstandard = feature_standard(dltborigin.values)
    dltbstandardfp = r"data/feature/dltb_standard.shp"
    outdata = pd.DataFrame(data=dltbstandard, columns=dltborigin.columns)
    outdata.to_csv(dltbstandardfp, index=None, encoding='utf-8-sig')

    feature = dltbstandard
    Comsidefp = r"data/feature/comside.txt"
    W = BuildMatrix(Comsidefp)
    noise = 0.4
    e = 0.0001
    layer = 2
    ncluster = 5
    labelfp = r"data/label/label.csv"

    for il in range(layer):
        newfeature = mDA(feature, noise, e, W)
        feature = newfeature.cpu().numpy()

    kmeans = KMeans(n_clusters=ncluster)
    label = kmeans.fit_predict(feature)
    LabelTocsv(label, labelfp)


if __name__ == '__main__':
    main()
