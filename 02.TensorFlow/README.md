# TensorFlow Learning Notes

- [Official Tutorial][318ec1de]

  [318ec1de]: https://www.tensorflow.org/get_started/get_started#tensorflow_core_tutorial "Official Tutorial"


## Part1_GetStart
### 0100_Install
>	安装TensorFlow

- 参考该文件夹下的文档基本就可以搞定
- [可以参考这篇博客][1590c02c]

  [1590c02c]: http://blog.csdn.net/chongtong/article/details/53905625?locationNum=6&fps=1 "博客"

### 0200_GetStart
> TensorFlow入门的一些demo，写在了jupyter里

- 0100_BasicDemo

	tensorflow的一些基本语法

- 0200_LinearModel

	使用tf的constant、Variable、placeholder等自己定义一个线性模型，并用tf.train API训练此模型

- 0300_tf.contrib.learn

	TensorFlow中有两种级别的API：

	1. TensorFlow Core：允许你进行完全的编程控制
	2. higher level API(方法名中包含contrib的):这些API是构建在TensorFlow Core之上的，你可以直接使用内置的模型

	这个demo中，直接使用内置的`Logistic Regression`模型创建一个`estimator`,并用其`fit`、`evaluate`方法进行训练和测试

- 0400_CUstomModel

	同样是使用higher level API,不同的是用`tf.contrib.learn.Estimator`构造一个自定义的model


## Part2_MNIST
### 0300_MnistForBeginner

- 0100_MNISTForBeginner

	自定义一个线性模型，然后用`tf.nn.softmax`分类MNIST

### 0400_MnistForExpert

- 0100_MNISTForExpert

	自定义`softmax`分类数据

- 0200_ConvNetModel

	两层卷积，两层全连接

### 0500_TensorFlowMechanics101

- 0100_tf.name_scope

	`name_scope`的一个demo，这个东西就类似于局部变量


## Part3_high-level_API
### 0600——tf.contrib.learnQuickstart

- 0100_NN

	使用`tf.contrib.learn`API构建一个神经网络，分类iris数据
