#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/12 11:23
# @Author  : Su.

from Word import Word
import jieba
import jieba.posseg as pseg
import pandas as pd
from collections import defaultdict
import os
import sys
sys.path.append("../")
reload(sys)
sys.setdefaultencoding("utf-8")

class senti_rule_model():
    def __init__(self):
        self.negative = []
        self.adverb = []
        self.questionMark = []
        self.rootPath = "E:\workout\data\senitment_data"
        self.wordtypeDict, self.wordfreqDict = self.UserDefineLibrary()
        for word in self.wordfreqDict.keys():
            jieba.add_word(str(word))
        self.initialize()

    # @staticmethod
    def initialize(self):
        self.negative.append("不")
        self.negative.append("伐")
        self.negative.append("都没")
        self.negative.append("不怎么")
        self.negative.append("不是")
        self.negative.append("没有")
        self.negative.append("好不")
        self.negative.append("并非")
        self.negative.append("不太")
        self.negative.append("很不")
        self.negative.append("不咋")
        self.negative.append("没什么")
        self.negative.append("没有什么")
        self.negative.append("都没有")
        self.negative.append("不咋")
        self.negative.append("还没")
        self.negative.append("太不")
        self.negative.append("并不")
        self.negative.append("不够")
        self.negative.append("也不")
        self.negative.append("我不太")

        self.adverb.append("非常")
        self.adverb.append("特别")
        self.adverb.append("很")
        self.adverb.append("狠")
        self.adverb.append("超级")
        self.adverb.append("太")
        self.adverb.append("好")
        self.adverb.append("真是太")

        self.questionMark.append("?")
        self.questionMark.append("？")
        self.questionMark.append("吗")
        self.questionMark.append("呢")
        self.questionMark.append("不")
        self.questionMark.append("点吧")

    # @staticmethod
    def UserDefineLibrary(self):
        dictPath = os.path.join(self.rootPath,"admindictdata.csv")
        data = pd.read_csv(dictPath, sep="\t", encoding="utf-8",
                           names=["word", "type", "freq"])

        wordtypeDict = defaultdict(str)
        wordfreqDict = defaultdict(int)

        for i in xrange(len(data["word"])):
            wordtypeDict.setdefault(data["word"][i],data["type"][i])
            wordfreqDict.setdefault(data["word"][i], data["freq"][i])
        return wordtypeDict, wordfreqDict

    def splitWord(self, content):

        segs = pseg.cut(str(content))
        result = []
        for word,type in segs:
            WORD = Word()
            if self.wordtypeDict.has_key(word):
                WORD.setword(word)
                WORD.settype(self.wordtypeDict[word])
                WORD.setfreq(self.wordfreqDict[word])
            else:
                WORD.setword(word)
                WORD.settype(type)
                result.append(WORD)
            # print "word ", word
            result.append(WORD)
        return result

    def sentiScoreDoc(self, tokens):
        posVec = []
        negVec = []
        posScore = 0.0
        negScore = 0.0
        for i in xrange(len(tokens)):
            term = tokens[i]
            preWord = Word()
            afterWord = Word()

            if (i -1 >= 0) and tokens[i-1] != None:
                if self.negative.__contains__(tokens[i-1].getword()) or\
                    self.adverb.__contains__(tokens[i-1].getword()):
                    preWord = tokens[i-1]
                else:
                    preWord.initialize("", "other", 1)
            else:
                preWord.initialize("", "other", 1)

            if (i +1 < len(tokens)) and tokens[i+1] != None:
                if self.questionMark.__contains__(tokens[i+1].getword()) :
                    afterWord = tokens[i+1]
                else:
                    afterWord.initialize("", "other", 1)
            else:
                afterWord.initialize("", "other", 1)

            if term.gettype().__contains__("pos-common"):
                posVec.append(term)
                # print preWord.getword()," term ", afterWord.getword()
                # print term.getword()
                posScore  += self.singleScore(preWord, term, afterWord)

            if term.gettype().__contains__("neg-common") or\
                term.gettype().__contains__("neg-need") or \
                term.gettype().__contains__("neg-game-dev-网络异常") or \
                    term.gettype().__contains__("neg-game-dev-登录问题") or \
                    term.gettype().__contains__("neg-game-dev-客户端问题") or \
                    term.gettype().__contains__("neg-game-dev-下载问题") or \
                    term.gettype().__contains__("neg-game-dev-bug反馈") or \
                    term.gettype().__contains__("neg-game-dev-更新差") or \
                    term.gettype().__contains__("gamecycle-3"):
                negVec.append(term)
                negScore += term.getfreq()

        score = 0.0
        # print "posScore",posScore
        # print "negScore",negScore
        if len(posVec)!=0 and len(negVec)!=0:
            score = posScore - negScore
        elif len(posVec)==0 and len(negVec)==0:
            return 0.0
        elif len(posVec)==0 and len(negVec)!=0:
            score = -(negScore/len(negVec))
        elif len(posVec)!=0 and len(negVec)==0:
            score = posScore/len(posVec)

        if score <= -2: score=-2
        elif score >=2: score=2

        if score >0:
            for WORD in tokens:
                if WORD.getword()=="?" or WORD.getword()=="？" or WORD.getword=="求":
                    return 0

        return score



    def singleScore(self, preWord, term, afterWord):
        if self.negative.__contains__(preWord.getword()) and self.questionMark.__contains__(afterWord.getword()):
            return 0
        elif self.adverb.__contains__(preWord.getword()) and self.questionMark.__contains__(afterWord.getword()):
            return 0
        elif preWord.getword() == "" and self.questionMark.__contains__(afterWord.getword()):
            return 0
        elif self.adverb.__contains__(preWord.getword()) and afterWord.getword()== "":
            return 1.5 * term.getfreq()
        elif self.negative.__contains__(preWord.getword()) and afterWord.getword() == "":
            return -term.getfreq()
        else:
            return term.getfreq()

