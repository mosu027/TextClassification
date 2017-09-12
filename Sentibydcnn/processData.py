#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/17 14:45
# @Author  : Su.


from collections import Counter
import itertools
import numpy as np
import re
import pandas as pd
import jieba



class processData():
    def __init__(self, trainPath, vocabmaxlen):
        self.trainPath = trainPath
        self.vocabmaxlen = vocabmaxlen



    def load_data_and_labels(self):
        """
        Loads data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        data = pd.read_csv(self.trainPath,sep="\t")

        x_text = list(data["text"])
        y = list(data["score"])
        test_size = int(np.round(0.10* len(y)))

        x_text = [list(jieba.cut(str(x))) for x in x_text]
        all_label = dict()
        for label in y:
            if not label in all_label:
                all_label[label] = len(all_label) + 1
        one_hot = np.identity(len(all_label))
        y = [one_hot[ all_label[label]-1 ] for label in y]

        return [x_text, y, test_size]

    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = self.vocabmaxlen
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            if len(sentence) <= sequence_length:
                num_padding = sequence_length - len(sentence)
                sentence.extend([padding_word] * num_padding)
                if len(sentence) == sequence_length:
                    padded_sentences.append(sentence)
                else:
                    print "error:",i
            else:
                padded_sentences.append(sentence[-sequence_length:])
        return padded_sentences

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        # vocabulary_inv=['<PAD/>', 'the', ....]
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        # Mapping from word to index
        # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def build_input_data(self, sentences, labels, vocabulary):
        """
        Maps sentences and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]

    def load_data(self):
        """
        Loads and preprocessed data
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels, test_size = self.load_data_and_labels()
        sentences_padded = self.pad_sentences(sentences)


        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)
        x, y = self.build_input_data(sentences_padded, labels, vocabulary)
        # print x[0]
        return [x, y, vocabulary, vocabulary_inv, test_size]


    def preprocess_dev_data(self, devdata):
        devdata = [list(jieba.cut(x)) for x in devdata]
        dev_sent_padded = self.pad_sentences(devdata)


        sentences, labels, test_size = self.load_data_and_labels()
        sentences_padded = self.pad_sentences(sentences)

        print "len_pad",len(sentences_padded)
        vocabulary, vocabulary_inv = self.build_vocab(sentences_padded)

        # dev_x = np.array([[vocabulary[word] for word in sentence] for sentence in dev_sent_padded])
        dev_x = []
        for sentence in dev_sent_padded:
            temp = []
            for word in sentence:
                try:
                    temp.append(vocabulary[word])
                except:
                    continue
            dev_x.append(temp)

        return np.array(dev_x)


    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = (batch_num + 1) * batch_size
                if end_index > data_size:
                    end_index = data_size
                    start_index = end_index - batch_size
                yield shuffled_data[start_index:end_index]