import numpy as np
import scipy.io as sio

labelsPath = 'F:/Deep Learning/eeg/dataset/health/y_test.mat'
logitsPath = 'F:/Deep Learning/eeg/dataset/health/logits.mat'

labels = sio.loadmat(labelsPath)['y_test'].reshape((-1,))
labels = labels - 1
logits = sio.loadmat(logitsPath)['data'][0,:,:]

predicts = np.argmax(logits, axis=1)

# True False组成的数组
results = (predicts == labels)
results = results.reshape((58, 15))
print(1*results)

negative = np.where(results==False)
# print('negative:')
# print(negative)
# print(negative[0].shape)
accuracy = 100 - len(negative[0]) / 870 * 100
print('accuracy:', accuracy)