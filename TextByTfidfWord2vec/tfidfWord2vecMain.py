#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2017/6/28 22:42
# @Author    :peng


import os
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import jieba

import tfidf_feature
import classifier
import word2vec_feature


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class TfidfWord2vecMoodel():

    def __init__(self):

        self.boolpreword2vec = False
        self.boolword2vec = True
        self.booltfidf = False
        self.rootPath = "E:\workout\data\senitment_data"
        self.resultpath = os.path.join(self.rootPath, "result_" + self.curTime() + ".csv")
        self.trainPath = os.path.join(self.rootPath, "train.csv")
        self.testPath = os.path.join(self.rootPath, "test.csv")
        self.dataAllPath = os.path.join(self.rootPath, "data20170908.csv")
        self.stopwords_path = "Conf/stopwords.txt"
        print self.trainPath
        self.word2vecModelPath = os.path.join(self.rootPath, "model/word2vecmodel.m")



    def load_data(self,path):
        data = pd.read_csv(path,sep="\t",encoding="utf-8")
        x = data["text"]
        y = data["score"]

        x = [list(jieba.cut(str(line))) for line in list(x)]
        return  x, y

    def load_train(self):
        return self.load_data(self.trainPath)

    def load_test(self):
        return  self.load_data(self.testPath)

    def load_dataAll(self):
        return  self.load_data(self.dataAllPath)


    def curTime(self):
        return str(datetime.datetime.now()).replace(" ", "")\
                                            .replace("-","")\
                                            .replace(":","")\
                                            .split(".")[0]



    def splitData(self, X, Y):

        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.4,
                                                random_state=1)
        return xtrain, xtest, ytrain, ytest



    def main(self):
        xtrain,ytrain = self.load_train()
        xtest, ytest = self.load_test()
        print list(set(ytrain))
        print list(set(ytest))

        if self.booltfidf == True:
            xtrain, xtest = tfidf_feature.tfidf_feature(xtrain, xtest, self.stopwords_path)
            classifier.trainModel(xtrain, xtest, ytrain, ytest)
        elif self.boolword2vec== True:
            important_words = None
            xtrain = word2vec_feature.word2vec_feature(xtrain, important_words, self.word2vecModelPath)
            xtest = word2vec_feature.word2vec_feature(xtest, important_words, self.word2vecModelPath)
            classifier.trainModel(xtrain, xtest, ytrain, ytest)
        elif self.boolpreword2vec == True:

            "train word2vecmodel..."
            data = self.load_dataAll()
            print len(data[0])
            word2vec_feature.train_word2vec_model(data[0], self.word2vecModelPath)



if __name__ == '__main__':
    TfidfWord2vecMoodel().main()


