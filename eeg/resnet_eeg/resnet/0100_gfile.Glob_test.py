import tensorflow as tf
import numpy as np

data_path = '../eeg/train*'
# 返回匹配data_path这个模式的文件名列表
data_files = tf.gfile.Glob(data_path)

print(data_files)   # ['..\\eeg\\train.mat', '..\\eeg\\train_labels.mat']