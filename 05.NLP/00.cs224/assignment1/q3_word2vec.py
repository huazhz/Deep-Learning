#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    # 不太懂为啥这么归一化，用每行的数除以这行所有数平方和的平方根
    N = x.shape[0]
    x_sum = np.sum(x ** 2, axis=1)
    x_sqrt = np.sqrt(x_sum).reshape((N, 1))
    x /= x_sqrt
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector
                 这应该指的是hidden layer的计算结果 [1,n]
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
                     pdf里的U是[n,V]的矩阵,as rows指的是[V,n]
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    # reshape成[1,n]
    predicted = predicted.reshape((1, -1))
    # 从[V,n]变成[n,V]
    outputVectors = np.transpose(outputVectors, (1, 0))

    # [1,V]
    score = np.dot(predicted, outputVectors)
    y_estimate, _ = softmax(score)
    cost = - np.log(y_estimate[0, target])
    dscore = y_estimate.copy()
    dscore[0, target] -= 1
    # [1,n]
    gradPred = np.dot(dscore, outputVectors.T)
    # [V,n]
    grad = np.dot(predicted.T, dscore).transpose((1, 0))

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Arguments:
    predicted -- numpy ndarray, predicted word vector
                 这应该指的是hidden layer的计算结果 [1,n]
    target -- integer, the index of the target word
              当前根据中心词所预测的词的索引值
    outputVectors -- "output" vectors (as rows) for all tokens
                     pdf里的U是[n,V]的矩阵,as rows指的是[V,n]
    dataset -- needed for negative sampling.
               dataset.sampleTokenIdx: 返回0~4之间的一个随机数，字典中共5个单词
               dataset.getRandomContext： 得到一个如下形式的映射
                 ('b', ['d', 'a', 'a', 'c', 'b', 'e', 'b', 'd', 'e', 'c'])
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    # [target, ...] 共11个数, 代表着以target词为中心的10个word pair
    indices.extend(getNegativeSamples(target, dataset, K))

    cost = 0.0
    gradPred = np.zeros_like(predicted)
    grad = np.zeros_like(outputVectors)
    V, n = outputVectors.shape

    sig_pos = sigmoid(np.dot(predicted, outputVectors[target, :].T))[0]  # float
    cost += -np.log(sig_pos)
    gradPred += (sig_pos - 1) * outputVectors[target, :]
    grad[target, :] += np.reshape((sig_pos - 1) * predicted, [n, ])

    for k in range(K):
        target_k = indices[k + 1]
        sigs_neg = sigmoid(np.dot(-predicted, outputVectors[target_k, :].T))[0]
        cost += -np.log(sigs_neg)
        gradPred += (1 - sigs_neg) * outputVectors[target_k, :]
        grad[target_k, :] += np.reshape((1 - sigs_neg) * predicted, [n,])

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    contextWords -- list of no more than 2*C strings, the context words
    'b', ['d', 'a', 'a', 'c', 'b', 'e', 'b', 'd', 'e', 'c']

    C -- integer, context size
    tokens -- a dictionary that maps words to their indices in
              the word vector list
              [("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
    inputVectors -- [V,n]  "input" word vectors (as rows) for all tokens
    outputVectors -- [V,n]  "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """
    # softmaxCostAndGradient(predicted, target, outputVectors, dataset)
    # total_cost就是将每一个context word的cost加起来
    # Vc = one-hot.dot(inputVectors) -> gradIn = np.dot(one-hot.T, dVc) dVc==gradPred

    V, n = inputVectors.shape
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    # 中心词在索引中的下标
    currentWordIndex = tokens[currentWord]
    # 上下文在索引中的下标列表
    contextWordsIndices = [tokens[contextWords[i]] for i in range(len(contextWords))]
    # 中心词的one-hot向量
    currentVec = np.zeros((1, V))
    currentVec[0, currentWordIndex] = 1

    # [1,n]
    predicted = np.dot(currentVec, inputVectors)
    gradPred = np.zeros(predicted.shape)
    for i in range(2 * C):
        target_i = contextWordsIndices[i]
        cost_i, gradPred_i, grad_i = \
            word2vecCostAndGradient(predicted=predicted, target=target_i,
                                    outputVectors=outputVectors, dataset=dataset)
        cost += cost_i
        gradPred += gradPred_i
        gradOut += grad_i
    gradIn = np.dot(currentVec.T, gradPred)

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Arguments/Return specifications: same as the skip-gram model

    cbow模型是根据contextWords预测currentWord

    将contextWords中的每个词转成one-hot向量，计算hidden layer结果，然后将结果平均
    用平均之后的结果得到最后的output
    inputVectors -- [V,n]  "input" word vectors (as rows) for all tokens
    outputVectors -- [V,n]  "output" word vectors (as rows) for all tokens

    softmaxCostAndGradient(predicted, target, outputVectors, dataset)

    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    V, n = inputVectors.shape

    # 中心词在索引中的下标
    currentWordIndex = tokens[currentWord]
    # 上下文在索引中的下标列表
    contextWordsIndices = [tokens[contextWords[i]] for i in range(len(contextWords))]
    # 上下文的one-hot向量
    contextWordsVec = np.zeros((2 * C, V))
    contextWordsVec[range(2 * C), contextWordsIndices] = 1

    # hidden layer的结果
    predicted = np.zeros((1, n))
    # 所有contextVec的平均值
    inputVec = np.zeros((1, V))
    for i in range(2 * C):
        inputVec += contextWordsVec[i, :]
        currentPred = np.dot(contextWordsVec[i, :], inputVectors)
        predicted += currentPred
    predicted /= (2 * C)
    inputVec /= (2 * C)

    cost, gradPred, gradOut = \
        word2vecCostAndGradient(predicted=predicted, target=currentWordIndex, outputVectors=outputVectors,
                                dataset=dataset)
    gradIn = np.dot(inputVec.T, gradPred)

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0

    # wordVectors.shape这个应该是dummy_vectors [10,3]
    grad = np.zeros(wordVectors.shape)
    # 10
    N = wordVectors.shape[0]
    # [5,3]
    inputVectors = wordVectors[:N // 2, :]
    # [5,3]
    outputVectors = wordVectors[N // 2:, :]
    for i in range(batchsize):
        # C=5
        C1 = random.randint(1, C)
        # 'b', ['d', 'a', 'a', 'c', 'b', 'e', 'b', 'd', 'e', 'c']
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        # tokens: [("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
        c, gradIn, gradOut = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N // 2, :] += gradIn / batchsize / denom
        grad[N // 2:, :] += gradOut / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    # 返回0~4之间的一个随机数
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    # 得到一个这种形式的映射：
    # ('b', ['d', 'a', 'a', 'c', 'b', 'e', 'b', 'd', 'e', 'c'])
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(
        f=lambda vec: word2vec_sgd_wrapper(
            word2vecModel=skipgram, tokens=dummy_tokens, wordVectors=vec,
            dataset=dataset, C=5, word2vecCostAndGradient=softmaxCostAndGradient),
        x=dummy_vectors
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors
    )

    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors
    )
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(
            cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors
    )

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient))


if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()
