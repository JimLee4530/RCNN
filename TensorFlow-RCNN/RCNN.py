import tensorflow as tf
import numpy as np
from read_data import *

class RCNN:
    def __init__(self, time, K, p, numclass, is_training):
        self.time = time
        self.K = K
        self.p = p
        self.numclass = numclass
        self.is_training = is_training

    def RCL(self, X):
        with tf.variable_scope("RCL") as scope:
            wr = tf.get_variable('weight_r', [3, 3, self.K, self.K],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # # biases = tf.get_variable('biases_r', [self.K],
            # #                          initializer=tf.random_normal_initializer())
            conv1 = tf.layers.conv2d(X, self.K, kernel_size=(3, 3),padding='same',reuse=None, name='rcl')
            rcl1 = tf.add(conv1, X)
            bn1 = tf.contrib.layers.batch_norm(rcl1)
            #
            conv2 = tf.layers.conv2d(bn1, self.K, kernel_size=(3, 3), padding='same', reuse=True, name='rcl')
            rcl2 = tf.add(conv2, X)
            bn2 = tf.contrib.layers.batch_norm(rcl2)
            #
            conv3 = tf.layers.conv2d(bn2, self.K, kernel_size=(3, 3),padding='same', reuse=True, name='rcl')
            rcl3 = tf.add(conv3, X)
            bn3 = tf.contrib.layers.batch_norm(rcl3)

            return bn3


    def buile_model(self, X, y):

        with tf.variable_scope('conv1'):

            conv1 = tf.contrib.layers.conv2d(X, 192, kernel_size= (5, 5))
            output = tf.contrib.layers.batch_norm(conv1)


        for i in range(4):
            with tf.variable_scope('recurrent_conv_{}'.format(i)):
                conv2 = tf.contrib.layers.conv2d(output, 192, (3, 3))
                bn = tf.contrib.layers.batch_norm(conv2)
                output = self.RCL(bn)
                if i == 1:
                    output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                if i != 3:
                    output = tf.nn.dropout(output, keep_prob=0.8)

        with tf.variable_scope('global_max_pooling'):
            H = np.shape(output)[1]
            W = np.shape(output)[2]
            output = tf.reshape(tf.nn.max_pool(output, ksize=[1, H, W, 1], strides=[1, H, W, 1], padding='SAME'), [-1, self.K])

        with tf.variable_scope('softmax'):
            logits = tf.contrib.layers.fully_connected(output, self.numclass)
            label = tf.one_hot(y, depth=self.numclass)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
            loss = tf.reduce_mean(entropy, name='loss')
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, y))) / tf.cast(tf.shape(logits)[0], tf.float32)

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            # tf.summary.histogram('histogram loss', loss)
            summary_op = tf.summary.merge_all()


        return loss, summary_op, acc, preds
