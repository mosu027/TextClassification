#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/7 15:39
# @Author  : Peng


import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import jieba
import pandas as pd
import sys
import w2vProcessData
from connecthive import connecthive


pos_dir = "./data/rt-polaritydata/rt-polarity.pos"
neg_dir = "./data/rt-polaritydata/rt-polarity.neg"

ch_pos_dir = "../Data/input/pos.xls"
ch_neg_dir = "../Data/input/neg.xls"

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", ch_pos_dir, "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", ch_neg_dir, "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/20170712120455", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# x_evluate = []


# print data[0]

x_evluate = []


argvs_lenght = len(sys.argv)

dataclass = w2vProcessData.ProcessData()

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    # x_raw, y_test = dataclass.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # y_test = np.argmax(y_test, axis=1)
    pass
else:
    # x_raw = ["a masterpiece four years in the making", "everything is off."]
    if argvs_lenght == 2:
        sentence = sys.argv[-1]
        x_evluate.append(sentence)
    else:
        x_evluate = [u"这东西挺好的",u"这太坑了",u"太差了",u"王者荣耀还什么5v5公平对战手游",
                     u"体验不好",u"这东西不坑",u"英雄伤害挺高的", u"吊炸了",u"吊炸天"]
        # x_evluate = data


    # y_test = y_evaluate


x_test = dataclass.getTestDataX(x_evluate)

print("\nEvaluating...\n")


# Evaluation
# ==================================================
checkpoint_file_root = os.path.join(FLAGS.checkpoint_dir, "checkpoints")
checkpoint_file = tf.train.latest_checkpoint(checkpoint_file_root)
print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = dataclass.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


print all_predictions
# predictions_human_readable = np.column_stack((np.array(data), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictiontocsv.csv")
# pd.DataFrame(predictions_human_readable).to_csv(out_path,index=False, sep="\t", header= False)
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:




