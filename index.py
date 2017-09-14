#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time      :2017/6/28 22:39
# @Author    :peng


# from TextByTfidfWord2vec import tfidfWord2vecMain
# model = tfidfWord2vecMain.TfidfWord2vecMoodel()
# model.main()


# from SentiByCNNWord2vec import eval
# eval.showResult()
#
# from Sentibydcnn import eval
# eval.showResult()
#
# from Sentibyrule import eval
# eval.main()

# from SentiByCNNIndex import eval
# eval.showResult()



from SentiByBasicLSTM import basicLSTM
basicLSTM.evaluate()
# basicLSTM.train()
# basicLSTM.lstm_predict("漂亮")


# string = '根本不给送货上门！白花钱了！'
# basicLSTM.lstm_predict(string)