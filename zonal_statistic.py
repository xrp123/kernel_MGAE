# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Version : python2.7

import pandas as pd
import numpy as np
import arcpy
import os
from arcpy import env
from arcpy.sa import *
import sys

env.overwriteOutput = True


def ZonalStatistic(zoneField, inValueRaster, ZonalFeature, ipoiname, outTable):
    """

    :param zoneField:
    :param inValueRaster:
    :param ZonalFeature:
    :param ipoiname:
    :param outTable:
    :return:
    """
    inValueRaster = "{}\\{}.tif".format(inValueRaster, ipoiname)
    outTable = "{}\\{}.dbf".format(outTable, ipoiname)
    arcpy.CheckOutExtension("Spatial")
    outZSaT = ZonalStatisticsAsTable(ZonalFeature, zoneField, inValueRaster, outTable, "NODATA", "MEAN")
    print ("{}.dbf is finished!".format(ipoiname))


def AddField(dbfFile, ipoiname, expression, poiname, JH_poiname):
    """

    :param dbfFile:
    :param ipoiname:
    :param expression:
    :param poiname:
    :param JH_poiname:
    :return:
    """
    dbfFile = "{}\\{}.dbf".format(dbfFile, ipoiname)
    index = poiname.index(ipoiname)
    arcpy.AddField_management(dbfFile, JH_poiname[index], "DOUBLE")
    arcpy.CalculateField_management(dbfFile, JH_poiname[index], expression)
    print ("{}.dbf is added!".format(ipoiname))


def JoinField(inField, joinField, ipoiname, poiname, JH_poiname, inData, dbfFile):
    """

    :param inField:
    :param joinField:
    :param ipoiname:
    :param poiname:
    :param JH_poiname:
    :param inData:
    :param dbfFile:
    :return:
    """
    index = poiname.index(ipoiname)
    fieldList = [JH_poiname[index]]
    dbfFile = "{}\\{}.dbf".format(dbfFile, ipoiname)
    arcpy.JoinField_management(inData, inField, dbfFile, joinField, fieldList)
    print("{}.dbf is Joined!".format(ipoiname))


if __name__ == "__main__":
    ipoi = ['Science_and_Education', 'Corporate_business_Factory', 'Food_and_Beverage_place', 'Hotel', 'Bank_Financial',
            'Tourism_attraction', 'Public_facility', 'Governmental_and_Public_organizations', 'Sports_Recreation',
            'Car_service', 'Shopping_mall', 'Transportation_facilities', 'Residence', 'Culture_Media']
    JH_poiname = ['Science', 'Corporate', 'Food', 'Hotel', 'Bank', 'Tourism', 'Public', 'Government', 'Sports', 'Car',
                  'Shopping', 'Transpor', 'Residence', 'Culture']
    inValueRaster = "data/kernel_density/raster"
    ZonalFeature = "data/feature/dltb.shp"
    outTable = "data/table"
    for ipoiname in ipoi:
        ZonalStatistic("FID", inValueRaster, ZonalFeature, ipoiname, outTable)
        dbfFile = outTable + "\\{}.dbf".format(ipoiname)
        expression = '[MEAN]'
        AddField(dbfFile, ipoiname, expression, ipoi, JH_poiname)

        inData = ZonalFeature
        inField = "FID"
        joinField = 'FID_'
        JoinField(inField, joinField, ipoiname, ipoi, JH_poiname, inData, outTable)
