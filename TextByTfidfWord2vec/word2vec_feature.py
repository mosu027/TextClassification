#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/11 15:45
# @Author  : Su.

from gensim.models import Word2Vec
import numpy as np


def train_word2vec_model(data, word2vecModelPath):
    print "Trainword2vecmodel..."
    model = Word2Vec(data, min_count=3,size=100)
    model.save(word2vecModelPath)




def word2vec_feature(data, important_word, word2vecModelPath):
    """
    word2vec feature
    """
    # model = Word2Vec.load_word2vec_format(word2vecModelPath, binary=False)
    model = Word2Vec.load(word2vecModelPath)
    feature = []
    empty_count = 0
    wordsize= 100
    for doc in data:
        temp_list = [0]*wordsize
        temp_count = 0
        # doc_list = doc.split(" ")
        for word in doc:
            word = word.decode("utf-8")
            try:
                if len(temp_list) == 0:
                    temp_list = model[word]
                    temp_count = 1
                else:
                    temp_list = [x+y for x, y in zip(temp_list, model[word])]
                    temp_count += 1
            except:
                pass
        if len(temp_list) == [0]*wordsize:
            empty_count += 1
        else:
            if temp_count == 0:
                temp_list = [x/1for x in temp_list]
            else:
                temp_list = [x / float(temp_count) for x in temp_list]
            feature.append(temp_list)
    # print "empty_count:", empty_count
    return np.array(feature)