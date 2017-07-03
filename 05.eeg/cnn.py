import argparse
import os.path
import sys
import time
import math
import scipy.io as sio
import numpy as np


import tensorflow as tf
from . import eeg

# Basic model parameters as external flags.
FLAGS = None

def loadEEGData():
    dataPath = 'F:/Deep Learning/05.eeg/dataset/sz/'
    X_train = sio.loadmat(dataPath+'X_train.mat')['X_train']
    X_test = sio.loadmat(dataPath+'X_test.mat')['X_test']
    y_train = sio.loadmat(dataPath+'y_train.mat')['y_train']
    y_test = sio.loadmat(dataPath+'y_test.mat')['y_test']

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
    images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 45, 31, 1], name='X-input')
    labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='y-input')

    with tf.Graph().as_default():
        logits = eeg.inference(
            images_placeholder,
            is_training=True,
            depth1=FLAGS.depth1,
            depth2=FLAGS.depth2,
            depth3=FLAGS.depth3,
            dense1_units=FLAGS.dense1,
            dense2_units=FLAGS.dense2,
            dropout=FLAGS.dropout)

        loss = eeg.loss(logits, labels_placeholder)

        train_step = eeg.training(loss, FLAGS.learning_rate)

        accuracy = eeg.evaluation(logits, labels_placeholder)

        merged = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        init = tf.global_variables_initializer()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        sess.run(init)

        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)

        iter_per_epoch = int(math.ceil(X_train.shape[0]/FLAGS.batch_size))
        for e in range(FLAGS.epochs):
            start_time = time.time()
            for i in range(iter_per_epoch):
                # 每100次迭代，测试并记录模型在验证集上的准确率
                if i % 100 == 0:
                    summary, acc = sess.run([merged, accuracy],
                                            feed_dict={images_placeholder:X_test, labels_placeholder:y_test})
                    validation_writer.add_summary(summary, global_step=e*iter_per_epoch+i)
                    print('Validation Accuracy at step %s: %s' % (e*iter_per_epoch+i, acc))
                else:
                    # generate indicies for the batch
                    # 取模是因为上面是上取整，有可能超出总样本数
                    start_idx = (i*FLAGS.batch_size)%X_train.shape[0]
                    idx = train_indicies[start_idx:start_idx+FLAGS. batch_size]

                    summary, _ = sess.run(
                        [merged, train_step],
                        feed_dict={X: X_train[idx,:],
                                   y: y_train[idx],
                                   is_training: True }
                    )

                    train_writer.add_summary(summary, global_step=e*iter_per_epoch+i)

            duration = time.time() - start_time
            print('The time span of 1 epoch: %s' % (duration))
            saver.save(sess, save_path=FLAGS.log_dir+'/model', global_step=(e+1)*iter_per_epoch)

        # test the model
        acc = sess.run(accuracy, feed_dict={X:X_test, y:y_test, is_training:False})
        print('Test Accuracy: %s' % (acc))

        train_writer.close()
        validation_writer.close()



# start training

saver = tf.train.Saver()



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
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
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
        default=48,
        help='The depth of second conv layer.'
    )
    parser.add_argument(
        '--depth1',
        type=int,
        default=64,
        help='The depth of third conv layer.'
    )
    parser.add_argument(
        '--dense1',
        type=int,
        default=512,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--dense2',
        type=int,
        default=58,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
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
        default='/tmp/eeg',
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
