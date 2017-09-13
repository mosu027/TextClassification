#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/9/11 15:09
# @Author  : Su.

import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils import result


def trainModel(xtrain, xtest, ytrain, ytest):
    classifiers = [
        # KNeighborsClassifier(3),
        # SVC(kernel="linear",  probability=True),
        # NuSVC(probability=True),
        # DecisionTreeClassifier(),
        RandomForestClassifier(),
        # AdaBoostClassifier(),
        # GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200,
        #          subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        #          min_samples_leaf=1, min_weight_fraction_leaf=0.,
        #          max_depth=5),
        # GradientBoostingClassifier(),
        # GaussianNB(),
        # LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis()
        ]

    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    for clf in classifiers:
        clf.fit(xtrain, ytrain)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(xtest)
        # acc = accuracy_score(ytest, train_predictions)
        # print("Accuracy: {:.4%}".format(acc))


        train_porb_predictions = clf.predict_proba(xtest)
        ll = log_loss(ytest, train_porb_predictions)
        print("Log Loss: {}".format(ll))

        # printResult(ytest, train_predictions)
        # result.printMultiResult(ytest, train_predictions)

        save_path = "doc/result.txt"
        desc = "sentiment by tfidf "
        result_str = result.printMultiResult(ytest, train_predictions)
        result.saveResult(save_path, desc, result_str)


        #
        # log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        # log = log.append(log_entry)


    print("=" * 30)


def predModel(trainX, trainY, testX):
    model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=250,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=6, min_impurity_split=1e-7, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto')
    model.fit(trainX, trainY)
    pred = model.predict_proba(testX)
    pred = [x[list(model.classes_).index(1)] for x in pred]

    return pred

