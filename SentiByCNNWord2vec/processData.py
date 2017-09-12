#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/6 19:28
# @Author  : Peng
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from gensim.models import Word2Vec
import multiprocessing
from sklearn.model_selection import train_test_split
import jieba

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7

input_length = 100
cpu_count = multiprocessing.cpu_count()


class ProcessData():

    def __init__(self, trainPath, w2vModelPath, dev_rate=0.1):

        self.tranPath = trainPath
        self.w2vModelPath = w2vModelPath
        self.vocabmaxlen = 100
        self.dev_rate = dev_rate



    def loadData(self):
        data = pd.read_csv(self.tranPath, sep="\t")
        trainX = data["text"]
        trainY = []
        for score in list(data["score"]):
            if score == 0:
                trainY.append([1, 0, 0])
            elif score == 1:
                trainY.append([0, 1, 0])
            else:
                trainY.append([0, 0, 1])
        return  trainX, np.array(trainY)


    # def loadData(self):
    #     data = pd.read_csv(self.posnegPath, sep="\t")
    #     data = data.dropna()
    #
    #     data = data.drop_duplicates()
    #
    #     posData = data[data.score > 0]
    #     negData = data[data.score < 0]
    #     neuData = pd.read_csv(self.neuPath)
    #
    #
    #     posData["text"] = posData.apply(lambda x: str(x["title"]) + " " + str(x["content"]), axis=1)
    #     negData["text"] = negData.apply(lambda x: str(x["title"]) + " " + str(x["content"]), axis=1)
    #
    #     trainX = np.concatenate((posData["text"], negData["text"], neuData["content"]))
    #
    #     posLabel = [[1, 0, 0] for _ in posData["text"]]
    #     negLabel = [[0, 1, 0] for _ in negData["text"]]
    #     neuLabel = [[0, 0, 1] for _ in neuData["content"]]
    #     y = np.concatenate([posLabel, negLabel, neuLabel], 0)
    #
    #     trainX = [self.replaceStrangeStr(x) for x in trainX]
    #     return trainX, y

    def replaceStrangeStr(self,rawstr):
        newstr = rawstr.replace("<br>"," ")\
                        .replace("【"," ")\
                        .replace("】"," ")
        return  newstr

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
    def create_dictionaries(self,model, trainX):

        gensim_dict = Dictionary()

        # print model.vocab.keys()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量


        data = []
        for sentence in trainX:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            data.append(new_txt)
        # return data

        # combined = parse_dataset(combined)
        trainX = data
        trainX = sequence.pad_sequences(trainX, maxlen=self.vocabmaxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, trainX

    def getTrainData(self):
        trainX,Y = self.loadData()
        trainX = [" ".join(list(jieba.cut(str(x)))) for x in trainX]

        w2vModel = Word2Vec.load(self.w2vModelPath)
        w2indx, w2vec, trainX = self.create_dictionaries(w2vModel,trainX)

        n_symbols = len(w2indx) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
        for word, index in w2indx.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = w2vec[word]
        x_train, x_test, y_train, y_test = train_test_split(trainX, Y, test_size=self.dev_rate)

        print "embedding_weights",len(embedding_weights),len(embedding_weights[0])

        embedding_weights = list(np.array(embedding_weights).astype(np.float32))
        embedding_weights = [list(x) for x in embedding_weights]



        return embedding_weights, x_train, x_test, y_train, y_test


    def getTestDataX(self, testX):
        testX = [" ".join(list(jieba.cut(str(x)))) for x in testX]
        w2vModel = Word2Vec.load(self.w2vModelPath)
        _, _, testX = self.create_dictionaries(w2vModel,testX)
        return testX






    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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


    def trainoWord2vecModel(self):

        trainX,_ = self.loadData()

        trainX = [list(jieba.cut(str(x))) for x in trainX]

        model = Word2Vec(size=100,
                         min_count=3,
                         window=3,
                         workers=4,
                         iter=2)
        model.build_vocab(trainX)
        model.train(trainX)
        # model = Word2Vec(trainX)
        model.save(self.w2vModelPath)
        # print model[u"不"]
        # for word,dist in model.similar_by_word(u"喜欢",10):
        #     print word,dist


        # print model[u"喜欢"]
        # print model.similar_by_word(u"好",10)
        # print model.similar_by_word(u"英雄",10)
        print "save word2vecmodel success!"

if __name__ == '__main__':
    data = ProcessData()
    data.trainoWord2vecModel()
    # w2vModel = Word2Vec.load(self.w2vModelPath)


