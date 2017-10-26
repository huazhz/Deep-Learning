# 下面总结了一些tf中容易出现的一些bug

1. 进行reduce操作的时候没有设置`keepdims=True`

	```
		ValueError: Dimensions must be equal, but are 5 and 64 for 'truediv' (op: 'RealDiv') with input shapes: [64,5], [64].
	```

	```
		# 正确代码：
		x_sum = tf.reduce_sum(x_exp, axis=1, keep_dims=True)
	```
2. 提示类型不一致，通常是在定义变量的时候没有指定好类型

	```
		# np定义数组的时候就指定astype
		b = tf.Variable(np.zeros([1, self.config.n_classes]).astype(np.float32), name='biases')
	```
