#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/13 14:06
# @Author  : Su.


import pandas as pd
import numpy as np


def loadtrainData(path):
    data = pd.read_csv(path, sep="\t")
    trainX = data["text"]
    trainY = []
    for score in list(data["score"]):
        if score == 0:
            trainY.append([1, 0, 0])
        elif score == 1:
            trainY.append([0, 1, 0])
        else:
            trainY.append([0, 0, 1])
    return trainX, np.array(trainY)



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
