import tensorflow as tf
from scipy import io as sio
import numpy as np

# resnet_ensemble1
DATA_PATH = 'E:/Resource/Deep Learning/eeg/resnet_fft/dataset/'


def loadEEGData(data_path=DATA_PATH):
    X_train = sio.loadmat(data_path + 'X_train.mat')['X_train']
    X_test = sio.loadmat(data_path + 'X_test.mat')['X_test']
    y_train = sio.loadmat(data_path + 'y_train.mat')['y_train']
    y_test = sio.loadmat(data_path + 'y_test.mat')['y_test']

    # labels是从1到58，改成0到57
    y_train = y_train - 1
    y_test = y_test - 1

    X_train = np.reshape(X_train, (-1, 32, 32, 1))
    y_train = np.reshape(y_train, (-1,))
    X_test = np.reshape(X_test, (-1, 32, 32, 1))
    y_test = np.reshape(y_test, (-1,))

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_test, y_test

def build_input(batch_size, mode):
    NUM_CLASS = 58
    X_train, y_train, X_test, y_test = loadEEGData()
    y_train -= 1
    y_test -= 1

    if mode == 'train':
        print('X_trian.shape', X_train.shape)
        print('y_train.shape', y_train.shape)

        examples = tf.convert_to_tensor(X_train, dtype=tf.float32)
        labels = tf.one_hot(indices=y_train, depth=NUM_CLASS)
    else:
        print('X_test.shape', X_test.shape)
        print('y_test.shape', y_test.shape)

        examples = tf.convert_to_tensor(X_test, dtype=tf.float32)
        labels = tf.one_hot(indices=y_test, depth=NUM_CLASS)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((examples, labels))
    print(dataset.output_shapes)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    next_examples, next_labels = iterator.get_next()
    return next_examples, next_labels