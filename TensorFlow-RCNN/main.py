from read_data import *
import numpy as np
import tensorflow as tf
import os
from RCNN import *
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
import os; os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    log_dir = './logdir'
    snapshot_interval = 10000
    snapshot_dir = './snapshot_dir'
    max_iter = 100000
    log_interval = 100

    lr = 0.0005

    file = 'data/train_32x32.mat'
    X_raw, y_raw = getData(filename=file)
    n_train = X_raw.shape[0]
    y_raw[y_raw == 10] = 0
    y_raw = np.reshape(y_raw, (n_train,))

    with tf.Session() as sess:
        X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None,))
        rcnn = RCNN(time=3, K=192, p=0.9, numclass=10, is_training=True)
        loss, summary_op, acc, _ = rcnn.buile_model(X, y)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots

        writer = tf.summary.FileWriter(log_dir, sess.graph)
        np.random.seed(0)
        loss_mean = 0
        acc_mean = 0
        start = datetime.datetime.now()
        for n_iter in range(max_iter):
            index = np.random.choice(n_train, 64, replace=True)
            image = X_raw[index]
            labels = y_raw[index]
            # print(image.shape)
            loss_batch, summary_op_batch, acc_batch, _ = sess.run([loss, summary_op, acc, optimizer], feed_dict={X:image, y:labels})
            loss_mean += loss_batch
            acc_mean += acc_batch
            if (n_iter + 1) % log_interval == 0 or (n_iter + 1) == max_iter:
                loss_mean = loss_mean/(log_interval*1.0)
                acc_mean = acc_mean/(log_interval*1.0)
                batch_time = datetime.datetime.now()
                print(
                    "time: {},iter = {}\n\tloss = {}, accuracy (cur) = {} ".format(batch_time - start, n_iter + 1, loss_mean,
                                                                                   acc_mean))
                loss_mean = 0
                acc_mean = 0

            writer.add_summary(summary_op_batch, global_step=n_iter)

            if (n_iter + 1) % snapshot_interval == 0 or (n_iter + 1) == max_iter:
                snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter + 1))
                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                print('snapshot saved to ' + snapshot_file)

    end = datetime.datetime.now()
    print("sum time: {}".format(end - start))
    writer.close()

if __name__ == '__main__':
    main()