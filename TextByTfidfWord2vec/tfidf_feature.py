#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2017/6/28 22:46
# @Author    :peng


import codecs
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer




def tfidf_feature(xtrain, xtest, stopwords_path):
    """
    tf-idf feature
    """
    xtrain = [" ".join(word) for word in xtrain]
    xtest = [" ".join(word) for word in xtest]
    stopwords = codecs.open(stopwords_path, 'r', encoding='utf-8').readlines()
    stopwords = [word.strip("\n") for word in stopwords]
    vectorizer_train = CountVectorizer(analyzer='word', stop_words=stopwords,min_df=5)
    count_train = vectorizer_train.fit_transform(xtrain)
    vectorizer_test = CountVectorizer(vocabulary=vectorizer_train.vocabulary_)
    count_test = vectorizer_test.fit_transform(xtest)

    transformer = TfidfTransformer()
    tfidf_train = transformer.fit(count_train).transform(count_train)
    tfidf_test = transformer.fit(count_test).transform(count_test)

    return tfidf_train.toarray(),tfidf_test.toarray()