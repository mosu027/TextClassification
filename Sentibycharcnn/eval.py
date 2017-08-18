#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/16 10:15
# @Author  : Su.


import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import csv
import jieba
import pandas as pd
import sys
sys.path.append("../")
from utils.connecthive import connecthive
import data_utils
import config


startdatetime = datetime.datetime.now()


def getRecentFile(rootPath):
    filenames = os.listdir(rootPath)
    recentfile = "0"
    for filename in filenames:
        print filename
        try:
            if int(filename) > int(recentfile):
                recentfile = filename
        except:
            continue
    return os.path.join(rootPath,recentfile)

recentFile = getRecentFile("runs/")
print recentFile

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", recentFile, "Checkpoint directory from training run")
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

cursor = connecthive("temp")
data = cursor.query("select chat_id, chat_content from inner_game.user_chat_log where part_game = 13 limit 1000")

x_evluate = [x[1].decode('utf8') for x in data]
x_raw = [x[0] for x in data]

print "conncet hive sucess!"
# x_evluate = data

print("\nEvaluating...\n")

execfile("config.py")
print config.model.th
dev_data = data_utils.Data(data_source=config.dev_data_source,
                alphabet=config.alphabet,
                l0=config.l0,
                batch_size=128,
                no_of_classes=3)


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
        input_x = graph.get_operation_by_name("Input-Layer/input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("Input-Layer/dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("OutputLayer/scores").outputs[0]
        predictions = tf.argmax(predictions, 1, name="predictions")

        # Generate batches for one epoch
        batches = dev_data.getDevBatchToIndices(x_raw)

        # Collect the predictions here
        all_predictions = []

        index = 0
        for x_test_batch in batches:
            index += 1
            print "Current index is ", index
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined

# print all_predictions


# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(data), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictiontocsv.csv")
pd.DataFrame(predictions_human_readable).to_csv(out_path,index=False, sep="\t", header= None)



print "Code cost time is " + str(datetime.datetime.now() - startdatetime)
print "End time is " + str(datetime.datetime.now())