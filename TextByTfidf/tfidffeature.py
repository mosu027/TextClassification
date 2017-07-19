#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2017/6/28 22:46
# @Author    :peng


import codecs
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import KernelPCA



class featureData():

    def __int__(self):
        self.trainPath = ""
        self.testPath = ""

    def loadData(self):
        trainData = pd.read_csv(self.trainPath)



    def tfidf_feature():
        """
        tf-idf feature
        """
        whole_data = load_files("./app/task/training/resources/data/")
        text_train, text_test, tag_train, tag_test \
                    = train_test_split(whole_data.data, whole_data.target, \
                                       test_size=0.3, random_state=30)
        stopwords = codecs.open('./app/conf/stopwords.txt', 'r', \
                                encoding='utf-8').readlines()
        stopwords = map(lambda word: word.strip("\n"), stopwords)

        vectorizer_train = CountVectorizer(analyzer='word', \
                                           stop_words=stopwords, \
                                           max_df=0.5)
        count_train = vectorizer_train.fit_transform(text_train)
        # joblib.dump(vectorizer_train, \
        #             "./app/task/training/resources/model_save/tfidf_model.m")
        vectorizer_test = CountVectorizer(vocabulary=vectorizer_train.vocabulary_)
        count_test = vectorizer_test.fit_transform(text_test)

        transformer = TfidfTransformer()
        tfidf_train = transformer.fit(count_train).transform(count_train)
        tfidf_test = transformer.fit(count_test).transform(count_test)

        # feature_train, feature_test = feature_select(
        #                                         tfidf_train.toarray(),
        #                                         tag_train,
        #                                         tfidf_test.toarray())

        # return feature_train, feature_test, tag_train, tag_test
        return tfidf_train,tfidf_test,tag_train,tag_test