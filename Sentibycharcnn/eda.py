#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/15 11:50
# @Author  : Su.


import pandas as pd



# datasource = "data/ag_news_csv/train.csv"
#
# data = pd.read_csv(datasource,sep= ",", quotechar='"',header = None)
# print data.head(5)
#
# print data.shape

def is_chinese(check_str):
    """
    judge if is chinese
    """
    for str_item in check_str:
        if u'\u4e00' <= str_item <= u'\u9fff':
            return True
    return False

def allchars():

    path = "../Data/data.csv"
    data = pd.read_csv(path,  sep = "\t", encoding="utf-8")
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    charall = list()
    for i in xrange(len(data["text"])):
        for char in data["text"][i]:
            if is_chinese(char) or alphabet.__contains__(char):
                if char not in charall:
                    charall.append(char)
    return charall

if __name__ == '__main__':
    allchars()