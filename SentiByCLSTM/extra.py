#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/15 14:58
# @Author  : Su.



import pandas as pd
import numpy as np
import os, sys

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Nadam
from keras import regularizers
from keras import backend as K
from keras.layers import MaxPooling1D, Conv1D, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
sys.setrecursionlimit(1000000)


rootPath = "/data/ws/data/biaozhu/contain_label"
trainPath = os.path.join(rootPath, "beibei.txt")

# 设置参数
# Embedding
maxlen = 128
embedding_size = 128
# Convolution
kernel_size = 5
filters = 64
pool_size = 4
# LSTM
lstm_output_size = 70
lstm_batch_size = 30
lstm_epochs = 10


# 加载训练文件
def loadfile():
    data = pd.read_csv(trainPath, sep="\001", names=['label', 'text'], encoding='utf-8')
    text = data["text"]
    Y = []
    for score in list(data["label"]):
        if score == 0:
            Y.append([1, 0])
        elif score == 1:
            Y.append([0, 1])

    return text, np.array(Y)


# 对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [document.replace('\n', '').split() for document in text]
    # text = [doc for doc in text if doc]
    return text


def generatedict(text):
    # 计算词典并保存
    # d2v_train = pd.concat([text], ignore_index = True)
    w = []  # 将所有词语整合在一起
    for i in text:
        w.extend(i)
    # for i in d2v_train:
    #     w.extend(i)
    dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数
    del w
    dict['id'] = list(range(1, len(dict)+1))
    # 这个 dict 需要保存下来
    # outputFile = modeldir + '/dict.data'
    # fw = open(outputFile, 'w')
    # pickle.dump(dict,fw)
    # fw.close()
    return dict


def word2index(text, dict):
    get_sent = lambda x: list(dict['id'][x])
    combine = pd.Series(text).apply(get_sent)
    print("Pad sequences (samples x time)")
    combine = list(sequence.pad_sequences(combine, maxlen=maxlen))
    return combine


def getdata(combine, Y):
    X = np.array(list(combine)) #全集
    # 改成三分类需要进行一定的调整，即把 y 转化为向量表示，比如第一类就是 [1,0,0]，第二类就是 [0,1,0]
    # Y = np_utils.to_categorical(np.array(list(y)))

    # 生成训练和测试集
    # random_state 为 1 则表示每次都固定，用于检验，不填或者填 0 为
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

    return x_train, y_train, x_test, y_test, X


def cnn_lstm(dict, x, y, xt, yt):
    model = Sequential()
    model.add(Embedding(len(dict)+1, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    # 这一步用来确定要分多少类，这个 1 表示 1 分类
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    print '模型构建完成'
    # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print "模型编译完成"
    hist = model.fit(x, y, batch_size=lstm_batch_size, epochs=lstm_epochs, verbose=1,
                     validation_split=0.1)
    print hist.history
    print "模型训练完成"
    # print ("保存模型")
    # yaml_string = model.to_yaml()
    # with open(modeldir + '/lstm.yml', 'w') as outfile:
    #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    # model.save_weights(modeldir + '/lstm.h5')
    print "测试集评估"
    score = model.evaluate(xt, yt, verbose=0)
    print "准确率:",score

    return model


# 训练入口函数
def train_lstm(dict,x,y,xt,yt):
  return cnn_lstm(dict,x,y,xt,yt)


#训练模型，并保存
def train():
    print 'Loading Data...'
    X,Y = loadfile()
    print 'Tokenising...'
    pn = tokenizer(X)
    print 'Generating Dict...'
    dict = generatedict(pn)
    print 'Word to Index...'
    pn = word2index(pn, dict)
    print 'Preparing data...'
    x,y,xt,yt,xa = getdata(pn, Y)
    print 'Model Stage...'
    # 这里训练全量模型
    model = train_lstm(dict, x, y, xt, yt)
    #print('Save Test Result...')
    #saveresult(model, xt, pn)
    print "Done"

    return model


# def loaddict():
#   fr = open(modeldir + '/dict.data')
#   dict = pickle.load(fr)
#   return dict


# def classify(text):
#     dict = loaddict()
#
#     with open(modeldir + '/lstm.yml', 'r') as f:
#         yaml_string = yaml.load(f)
#     model = model_from_yaml(yaml_string)
#     model.load_weights(modeldir + '/lstm.h5')
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     textvec = []
#     for item in text:
#         # 如果不在词典里，则直接丢弃（因为出现的次数也非常少，不考虑）
#         if item in dict['id']:
#             textvec.append(dict['id'][item])
#     textvec = pd.Series(textvec)
#     textvec = sequence.pad_sequences([textvec], maxlen=maxlen)
#     # 概率
#     prob = model.predict(textvec, verbose=0)
#     proba = model.predict_proba(textvec, verbose=0)
#     print "The preidction is : ", prob


if __name__=='__main__':

    model = train()

    # test = ['麻烦 看到 回答 一下 谢谢']
    # model.classify(test)

