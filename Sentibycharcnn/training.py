import datetime
import time
import numpy as np
import tensorflow as tf
import sys
import os
import config
from data_utils import Data
from char_cnn import CharConvNet

learning_rate = 0.001
if __name__ == '__main__':
    print 'start...'
    execfile("config.py")
    print config.model.th
    print 'end...'
    print "Loading data ....",
    train_data = Data(data_source = config.train_data_source,
                      alphabet = config.alphabet,
                      l0 = config.l0,
                      batch_size = config.batch_size,
                      no_of_classes = config.no_of_classes)
    train_data.loadData()
    dev_data = Data(data_source = config.dev_data_source,
                      alphabet = config.alphabet,
                      l0 = config.l0,
                      batch_size = config.batch_size,
                      no_of_classes = config.no_of_classes)
    
    dev_data.loadData()

    num_batches_per_epoch = int(train_data.getLength() / config.batch_size) + 1
    num_batch_dev = dev_data.getLength()
    print "Loaded"

    print "Training ===>"

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement = True,
                                      log_device_placement = False)

        sess = tf.Session(config = session_conf)

        with sess.as_default():

            char_cnn = CharConvNet(conv_layers = config.model.conv_layers,
                                   fully_layers = config.model.fully_connected_layers,
                                   l0 = config.l0,
                                   alphabet_size = config.alphabet_size,
                                   no_of_classes = config.no_of_classes,
                                   th = config.model.th)

            global_step = tf.Variable(0, trainable=False)
            
            # boundaries = []
            # br = config.training.base_rate
            # values = [br]
            # for i in range(1, 10):
            #     values.append(br / (2 ** i))
            #     boundaries.append(15000 * i)
            # values.append(br / (2 ** (i + 1)))
            # print(values)
            # print(boundaries)
            # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

            #learning_rate = tf.train.exponential_decay(config.training.base_rate,
            #                                           global_step,
            #                                           config.training.decay_step,
            #                                           config.training.decay_rate,
            #                                           staircase=True)

            #optimizer = tf.train.MomentumOptimizer(learning_rate, config.training.momentum)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(char_cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
                    
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            def curTime():
                return str(datetime.datetime.now()).replace(" ", "") \
                    .replace("-", "") \
                    .replace(":", "") \
                    .split(".")[0]

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", curTime()))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", char_cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", char_cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            sess.run(tf.global_variables_initializer())
            
                                     
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  char_cnn.input_x: x_batch,
                  char_cnn.input_y: y_batch,
                  char_cnn.dropout_keep_prob: config.training.p
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op,
                     global_step,
                     train_summary_op,
                     char_cnn.loss,
                     char_cnn.accuracy],
                    feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  char_cnn.input_x: x_batch,
                  char_cnn.input_y: y_batch,
                  char_cnn.dropout_keep_prob: 1.0 # Disable dropout
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step,
                     dev_summary_op,
                     char_cnn.loss,
                     char_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # train_data.shuffleData()
            startdatetime = datetime.datetime.now()

            for e in range(config.training.epoches):
                print "Current epoch is ",e
                train_data.shuffleData()
                for k in range(num_batches_per_epoch):

                    batch_x, batch_y = train_data.getBatchToIndices(k)
                    train_step(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    
                    if current_step % config.training.evaluate_every == 0:
                        xin, yin = dev_data.getBatchToIndices()
                        print("\nEvaluation:")
                        dev_step(xin, yin, writer=dev_summary_writer)
                        print("")
                        print "Model cost time is " + str(datetime.datetime.now() - startdatetime)
                        
                    if current_step % config.training.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))               

