# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python3.6

import math
import geopandas as gp
import numpy as np
import pandas as pd
import gdal
import pyproj
import ogr
import time

import torch
from shapely.geometry import LineString


def pixel2geo(GeoTransform, xpixel, ypixel, cellsize):
    """
    将平面坐标信息转换为实际平面坐标
    :param GeoTransform:
    :param xpixel:
    :param ypixel:
    :param cellsize:
    :return:
    """
    xgeo = GeoTransform[0] + xpixel * GeoTransform[1] + ypixel * GeoTransform[2] + cellsize / 2
    ygeo = GeoTransform[3] + xpixel * GeoTransform[4] + ypixel * GeoTransform[5] - cellsize / 2
    return xgeo, ygeo


def poi2pd(poidata):
    """
    从整体的POI列表中提取出tag，xgeo和ygeo
    :param poidata:
    :return:
    """
    temp = poidata.loc[:, ['tag', 'xgeo', 'ygeo']]
    xgeo = np.array(temp.loc[:, 'xgeo'])
    ygeo = np.array(temp.loc[:, 'ygeo'])
    length = len(xgeo)
    return xgeo, ygeo, length


def caldistance(xgeo, ygeo, poix, poiy):
    """
    计算POI点到栅格中心点的距离
    :param xgeo:
    :param ygeo:
    :param poix:
    :param poiy:
    :return:
    """
    return math.sqrt(math.pow((xgeo - poix), 2) + math.pow((ygeo - poiy), 2))


def existbarry(x_pixel, y_pixel, x_point, y_point, lineset):
    """
    判断POI点与栅格中心点之间知否存在阻隔
    :param x_pixel:
    :param y_pixel:
    :param x_point:
    :param y_point:
    :param lineset:
    :return:
    """
    line = LineString([(x_pixel, y_pixel), (x_point, y_point)])
    linebarry = np.array(lineset.intersects(line))
    if True in linebarry:
        return False
    else:
        return True


def kernel_density(ref_shp_fp, poilist, out_fp, cellsize, radius, lineset):
    """
    计算核密度并导出
    :param ref_shp_fp:
    :param poilist:
    :param out_fp:
    :param cellsize:
    :param radius:
    :param lineset:
    :return:
    """
    # 根据掩膜shp建立最大面积栅格tif
    ref_shp = ogr.Open(ref_shp_fp)
    m_layer = ref_shp.GetLayer()
    extent = m_layer.GetExtent()
    Xmin = extent[0]
    Xmax = extent[1]
    Ymin = extent[2]
    Ymax = extent[3]
    rows = int((Ymax - Ymin) / cellsize)
    cols = int((Xmax - Xmin) / cellsize)
    GeoTransform = [Xmin, cellsize, 0, Ymax, 0, -cellsize]
    target_ds = gdal.GetDriverByName('GTiff').Create(out_fp, xsize=cols, ysize=rows, bands=1,
                                                     eType=gdal.GDT_Float32)
    target_ds.SetGeoTransform(GeoTransform)
    pro = m_layer.GetSpatialRef()
    target_ds.SetProjection(str(pro))

    # 获取poi数据，提取数据坐标列表
    poi_xgeolist, poi_ygeolist, poilength = poi2pd(poilist)

    # 遍历每个POI点，计算密度值
    datarray = np.zeros((rows, cols))
    print("共{}个POI点".format(poilength))
    for ipoiindex in range(poilength):
        stime = time.time()
        poi_xgeo = poi_xgeolist[ipoiindex]
        poi_ygeo = poi_ygeolist[ipoiindex]
        poi_xpixel = int((poi_xgeo - Xmin) / cellsize)
        poi_ypixel = int((Ymax - poi_ygeo) / cellsize)
        xpixel_min = poi_xpixel - (int(radius / cellsize)) - 1 if (poi_xpixel - (int(radius / cellsize)) - 1) > 0 else 0
        xpixel_max = poi_xpixel + (int(radius / cellsize)) + 1 if (poi_xpixel + (
            int(radius / cellsize)) + 1) < cols else cols
        ypixel_min = poi_ypixel - (int(radius / cellsize)) - 1 if (poi_ypixel - (int(radius / cellsize)) - 1) > 0 else 0
        ypixel_max = poi_ypixel + (int(radius / cellsize)) + 1 if (poi_ypixel + (
            int(radius / cellsize)) + 1) < rows else rows
        for y in range(ypixel_min, ypixel_max):
            for x in range(xpixel_min, xpixel_max):
                xgeo, ygeo = pixel2geo(GeoTransform, x, y, cellsize)
                distance = caldistance(xgeo, ygeo, poi_xgeo, poi_ygeo)
                # 判断在半径内
                if distance < radius:
                    # 判断是否有交叉
                    if existbarry(xgeo, ygeo, poi_xgeo, poi_ygeo, lineset):
                        D_value = (3 * (1 - (distance / radius) ** 2) ** 2) / (math.pi * (radius ** 2))
                        datarray[y][x] += D_value
        ytime = time.time() - stime
        print("poi:{},{:.2}s".format(ipoiindex, ytime))

    # 对栅格tif进行赋值
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(1)
    band.WriteArray(datarray)
    band.FlushCache()


def main():
    ref_shp_fp = r"data/kernel_density/ref.shp"  # the extent of study area
    linefp = r"data/kernel_density/barry.shp"  # barrier (main roads and river)
    poi_fp = r"data/kernel_density/poi.txt"  # poi
    cellsize = 10
    radius = 100
    print("处理范围：{}".format(ref_shp_fp))
    print("障碍路线：{}".format(linefp))
    print("POI:{}".format(poi_fp))
    print("搜索半径：{}".format(radius))
    print("栅格大小：{}".format(cellsize))

    poi = pd.read_csv(poi_fp)
    lineset = gp.read_file(linefp)

    # get all tag of poi
    poi_tag = poi.loc[:, 'tag'].tolist()
    poi_tag = list(set(poi_tag))

    for ipoitag in poi_tag:
        print(r"POI类别：{} 开始".format(ipoitag))
        startime = time.time()
        out_fp = r"data/kernel_density/raster/{}.tif".format(ipoitag)
        tagpoi = poi[poi['tag'].isin([ipoitag])]
        kernel_density(ref_shp_fp, tagpoi, out_fp, cellsize, radius, lineset)
        print(r"POI类别：{}，用时：{}s".format(ipoitag, int(time.time() - startime)))


if __name__ == "__main__":
    main()
