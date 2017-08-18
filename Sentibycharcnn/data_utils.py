import numpy as np
# import tensorflow as tf
import csv
import re
import pandas as pd

class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet,
                 l0=1014,
                 batch_size=128,
                 no_of_classes=3):
        
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source


    def loadData(self):
        data = []
        # with open(self.data_source, 'rb') as f:
        #     rdr = csv.reader(f, delimiter='\t')
        #     for row in rdr:
        #         txt = ""
        #         for s in row[1:]:
        #             txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
        #         data.append ((int(row[0]), txt))
        rawdata = pd.read_csv(self.data_source, sep="\t", encoding="utf-8")
        # print rawdata.head()
        for i in xrange(len(rawdata)):
            try:
                txt = rawdata["text"][i]
                data.append((int(rawdata["score"][i]), txt))
            except:
                print "errorline:", i
                break




        self.data = np.array(data)
        self.shuffled_data = self.data
        # print self.data[0][0]

    def shuffleData(self):
        data_size = len(self.data)
        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]         

    # def getBatch(self, batch_num=0):
    #     data_size = len(self.data)
    #     start_index = batch_num * self.batch_size
    #     end_index = min((batch_num + 1) * self.batch_size, data_size)
    #     return self.shuffled_data[start_index:end_index]

    def getBatchToIndices(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        # print batch_indices[0]    
        return np.asarray(batch_indices, dtype='int64'), classes



    def getDevBatchToIndices(self, hiveData):
        data_size = len(hiveData)


        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.batch_size
            end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
            batch_texts = hiveData[start_index:end_index]
            batch_indices = []

            for s in batch_texts:
                batch_indices.append(self.strToIndexs(s))
            yield batch_indices


    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, n):
            c = s[i]
            if c in self.dict:
                str2idx[i] = self.dict[c]
        return str2idx

    def getLength(self):
        return len(self.data)

if __name__ == '__main__':
    data = Data("../Data/data.csv")
##    E = np.eye(4)
##    img = np.zeros((4, 15))
##    idxs = data.strToIndexs('aghgbccdahbaml')
##    print idxs
    
    data.loadData()
    data.getBatchToIndices(0)
    # with open("test.vec", "w") as fo:
    #     for i in range(data.getLength()):
    #         c = data.data[i][0]
    #         txt = data.data[i][1]
    #         vec =  ",".join(map(str, data.strToIndexs(txt)))
            
    #         fo.write("{}\t{}\n".format(c, vec))

##    for i in range(3):
##        data.shuffleData()
##        batch_x, batch_y = data.getBatchToIndices()
##        print batch_x[0], batch_y[0]
