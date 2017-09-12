#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/12 11:24
# @Author  : Su.

import pandas as pd
import os
import senti_rule
from utils import result

def load_testData(path):
    data = pd.read_csv(path, sep="\t")
    return data["text"], data["score"]

def evaluate_testData(xtest, ytest):
    ypred = []
    model = senti_rule.senti_rule_model()
    index = 0
    for text in xtest:
        index +=1
        if index%1000==0:
            print  "index:",index


        tokens = model.splitWord(text)
        score = model.sentiScoreDoc(tokens)
        if score < 0:
            ypred.append(1)
        elif score >0:
            ypred.append(0)
        else:
            ypred.append(2)

    result.printMultiResult(ytest, ypred)

def main():
    rootPath = "E:\workout\data\senitment_data"
    testDataPath = os.path.join(rootPath, "test.csv")
    xtest, ytest = load_testData(testDataPath)
    evaluate_testData(xtest, ytest)


if __name__ == '__main__':
    main()

