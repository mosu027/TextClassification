#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/14 11:22
# @Author  : Su.

import os
import pandas as pd

rootPath = "E:\workout\data\senitment_data"

trainPath = os.path.join(rootPath, "train.csv")
testPath = os.path.join(rootPath, "test.csv")

def eda(path):
    data =  pd.read_csv(path, sep="\t",dtype=str)
    print "num is ", len(data["score"])
    print "pos num is ", len(data[data["score"]=="0"])
    print "neg num is ", len(data[data["score"] == "1"])
    print "neu num is ", len(data[data["score"] == "2"])

if __name__ == '__main__':
    print "train......"
    eda(trainPath)
    print "test......"
    eda(testPath)
