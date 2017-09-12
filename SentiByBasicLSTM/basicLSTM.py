#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/12 10:03
# @Author  : Su.

import pandas as pd
import numpy as np
import jieba
import yaml
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import multiprocessing
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from sklearn.model_selection import train_test_split


# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()

class SentiByLSTM():
    def __init__(self):
        self.posPath = 'Data/input/neg.xls'
        self.negPath = 'Data/input/pos.xls'



    #加载训练文件
    def loadfile(self):
        neg=pd.read_excel(self.posPath, header=None,index=None)
        pos=pd.read_excel(self.negPath, header=None,index=None)

        combined=np.concatenate((pos[0], neg[0]))
        y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

        return combined,y

    #对句子经行分词，并去掉换行符
    def tokenizer(self, text):
        ''' Simple Parser converting each document to lower-case, then
            removing the breaks for new lines and finally splitting on the
            whitespace
        '''
        text = [jieba.lcut(document.replace('\n', '')) for document in text]
        return text

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引

    def create_dictionaries(self, model=None, combined=None):
        ''' Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries

        '''
        if (combined is not None) and (model is not None):
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(),
                                allow_update=True)
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
            w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

            # print w2indx

            def parse_dataset(combined):
                ''' Words become integers
                '''
                data = []
                for sentence in combined:
                    new_txt = []
                    for word in sentence:
                        try:
                            new_txt.append(w2indx[word])
                        except:
                            new_txt.append(0)
                    data.append(new_txt)
                return data

            combined = parse_dataset(combined)
            combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
            return w2indx, w2vec, combined
        else:
            print 'No data provided...'

    def word2vec_train(self, combined):

        model = Word2Vec(size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         iter=n_iterations)
        model.build_vocab(combined)

        # model.train(combined, total_examples = len(combined))
        # model.save('lstm_data/Word2vec_model.pkl')
        index_dict, word_vectors, combined = self.create_dictionaries(model=model, combined=combined)
        print combined[0][0]
        return index_dict, word_vectors, combined

    def get_data(self, index_dict, word_vectors, combined, y):

        n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        print x_train.shape, y_train.shape
        print embedding_weights
        return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

    ##定义网络结构
    def train_lstm(self, n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
        print 'Defining a Simple Keras Model...'
        model = Sequential()  # or Graph or whatever
        model.add(Embedding(output_dim=vocab_dim,
                            input_dim=n_symbols,
                            mask_zero=True,
                            weights=[embedding_weights],
                            input_length=input_length))  # Adding Input Length
        model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print embedding_weights
        print 'Compiling the Model...'
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        print "Train..."
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1,
                  validation_data=(x_test, y_test))

        print "Evaluate..."
        score = model.evaluate(x_test, y_test,
                               batch_size=batch_size)

        # yaml_string = model.to_yaml()
        # with open('lstm_data/lstm.yml', 'w') as outfile:
        #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        # model.save_weights('lstm_data/lstm.h5')
        print 'Test score:', score



    def main(self):

        print 'Loading Data...'
        combined, y = self.loadfile()
        print len(combined), len(y)
        print 'Tokenising...'

        combined = self.tokenizer(combined)
        print combined[0]
        print combined[1]

        # combined = [" ".join(x) for x in combined]
        print 'Training a Word2vec model...'
        index_dict, word_vectors, combined = self.word2vec_train(combined)
        print 'Setting up Arrays for Keras Embedding Layer...'
        n_symbols, embedding_weights, x_train, y_train, x_test, y_test = self.get_data(index_dict, word_vectors, combined, y)
        print x_train.shape, y_train.shape
        self.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)



