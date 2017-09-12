#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/12 11:45
# @Author  : Su.



class Word():
    word = ""
    type = ""
    freq = 0


    def __init__(self):
        self.word = ""
        self.type = ""
        self.freq = 0

    def initialize(self,word,type,freq):
        self.word = word
        self.type = type
        self.freq = freq


    def setword(self, word):
        self.word = word

    def getword(self):
        return self.word

    def settype(self, type):
        self.type = type

    def gettype(self):
        return self.type

    def setfreq(self, freq):
        self.freq = freq

    def getfreq(self):
        return self.freq
