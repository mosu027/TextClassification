#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/17 16:14
# @Author  : Su.

import os
import tensorflow as tf
import processData
import numpy as np


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



trainPath = "../Data/data.csv"
sentence_length = 100


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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

# data = pd.read_csv("../Data/chatcontenttest.csv",header=None,names=["id","content"])
# x_evaluate = list(data["content"])
# x_evluate = data

x_evaluate = [u"这游戏玩的不爽",u"这皮肤太好看了",u"这英雄特别叼",u"这英雄伤害不高",u"嗯嗯",u"这样啊"]



datapreprocess = processData.processData(trainPath,sentence_length)


x_test = datapreprocess.preprocess_dev_data(x_evaluate)

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
        dropout_keep_prob = graph.get_operation_by_name("dropout").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("predictions").outputs[0]

        # Generate batches for one epoch

        batches = datapreprocess.batch_iter(list(x_test), FLAGS.batch_size, 1)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


print all_predictions