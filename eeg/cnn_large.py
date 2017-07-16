import argparse
import math
import sys
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf

from model import eeg_large as eeg

# Basic model parameters as external flags.
FLAGS = None

def loadEEGData():
    dataPath = 'F:/Deep Learning/eeg/dataset/health2/'
    X_train = sio.loadmat(dataPath+'X_train.mat')['X_train']
    X_test = sio.loadmat(dataPath+'X_test.mat')['X_test']
    y_train = sio.loadmat(dataPath+'y_train.mat')['y_train']
    y_test = sio.loadmat(dataPath+'y_test.mat')['y_test']

    # labels是从1到58，改成0到57
    y_train = y_train - 1
    y_test = y_test - 1

    X_train = np.reshape(X_train, (-1, 80, 31, 1))
    y_train = np.reshape(y_train, (-1,))
    X_test = np.reshape(X_test, (-1, 80, 31, 1))
    y_test = np.reshape(y_test, (-1,))

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    print('X_test.shape:', X_test.shape)
    print('y_test.shape:', y_test.shape)

    return X_train, y_train, X_test, y_test


# def placeholder_inputs(batch_size):
#     """Generate placeholder variables to represent the input tensors.
#
#     Args:
#       batch_size: The batch size will be baked into both placeholders.
#
#     Returns:
#       images_placeholder: Images placeholder.
#       labels_placeholder: Labels placeholder.
#     """
#     images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
#                                                            eeg.IMAGE_PIXELS))
#     labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
#     return images_placeholder, labels_placeholder


def run_training():
    X_train, y_train, X_test, y_test = loadEEGData()

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 80, 31, 1], name='images-input')
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='y-input')
        is_training = tf.placeholder(dtype=tf.bool)

        # logits = eeg.inference(
        #     images_placeholder=images_placeholder,
        #     is_training=is_training,
        #     depth1=FLAGS.depth1,
        #     depth2=FLAGS.depth2,
        #     depth3=FLAGS.depth3,
        #     dense1_units=FLAGS.dense1,
        #     dense2_units=FLAGS.dense2,
        #     dropout_rate=FLAGS.dropout)

        logits = eeg.inference(
            images_placeholder=images_placeholder,
            is_training=is_training,
            depth1=FLAGS.depth1,
            depth2=FLAGS.depth2,
            depth3=FLAGS.depth3,
            dense1_units=FLAGS.dense1,
            dense2_units=FLAGS.dense2,
            dropout_rate=FLAGS.dropout)

        loss = eeg.loss(logits, labels_placeholder)

        train_step = eeg.training(loss, FLAGS.learning_rate, FLAGS.learning_rate_decay)

        accuracy = eeg.evaluation(logits, labels_placeholder)

        merged = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        init = tf.global_variables_initializer()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        sess.run(init)

        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)

        # record the max test accuracy every epoch
        max_test_accuracy = 0
        iter_per_epoch = int(math.ceil(X_train.shape[0]/FLAGS.batch_size))
        for e in range(FLAGS.epochs):
            start_time = time.time()

            for i in range(iter_per_epoch):

                # generate indicies for the batch
                # 取模是因为上面是上取整，有可能超出总样本数
                start_idx = (i*FLAGS.batch_size)%X_train.shape[0]
                idx = train_indicies[start_idx:start_idx+FLAGS. batch_size]

                summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={
                        images_placeholder: X_train[idx,:],
                        labels_placeholder: y_train[idx],
                        is_training:True
                    }
                )

                train_writer.add_summary(summary, global_step=e*iter_per_epoch+i)

            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={
                                        images_placeholder:X_test,
                                        labels_placeholder:y_test,
                                        is_training: False
                                    })
            test_writer.add_summary(summary, global_step=e*iter_per_epoch)
            print('Test accuracy at epoch %s: %s' % (e, acc))

            if acc > max_test_accuracy:
                max_test_accuracy = acc
                print('Max test accuracy: %s' % (max_test_accuracy))

            duration = time.time() - start_time
            print('The time span of 1 epoch: %s' % (duration))
            saver.save(sess, save_path=FLAGS.log_dir+'/model', global_step=(e+1)*iter_per_epoch)

        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.005,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.9,
        help='Exponential decay learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        # default=50,
        default=32,
        help='Number of batch size.'
    )
    parser.add_argument(
        '--depth1',
        type=int,
        default=32,
        help='The depth of first conv layer.'
    )
    parser.add_argument(
        '--depth2',
        type=int,
        # default=48,
        default=64,
        help='The depth of second conv layer.'
    )
    parser.add_argument(
        '--depth3',
        type=int,
        # default=64,
        default=128,
        help='The depth of third conv layer.'
    )
    parser.add_argument(
        '--dense1',
        type=int,
        default=1024,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--dense2',
        type=int,
        default=58,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default='0.5',
        help='Dropout rate.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='d:/tmp2/eeg_dp.5_deeper',
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
