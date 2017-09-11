#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/11 15:09
# @Author  : Su.


from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss



def printResult(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4%}".format(acc))

    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    print   "Precision:", precision
    print   "Recall:", recall
    print   "f1_score:", f1_score
    print   "confusion_matrix:"
    print   confusion_matrix

    resultStr = "Precision: " + str(precision) +"\n" + \
                "Recall: " + str(recall) + "\n" + \
                "f1_score: " + str(f1_score) +"\n" + \
                "confusion_matrix" + "\n" +\
                str(confusion_matrix) + "\n"
    return resultStr

def printMultiResult(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4%}".format(acc))

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    print   "confusion_matrix:"
    print   confusion_matrix

    resultStr = "confusion_matrix" + "\n" +\
                str(confusion_matrix) + "\n"
    return resultStr



def saveResult(savePath, time, timeconsuming, method, param, description, result):

    with open(savePath, "a+") as f:
        time = "Time: " + time + "\n"
        f.write(time)
        timeconsuming = "Timeconsuming: " + str(timeconsuming) + "\n"
        f.write(timeconsuming)
        method = "Method: " + method + "\n"
        f.write(method)
        param = "parma: " + param + "\n"
        f.write(param)
        description = "description: " + description + "\n"
        f.write(description)
        result = "result: " + result + "\n"
        f.write(result)

    print "save result success!"