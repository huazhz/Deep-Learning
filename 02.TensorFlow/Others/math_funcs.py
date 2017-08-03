import tensorflow as tf

a = tf.constant(1, shape=[1,3,2,2])
b = tf.constant(3, shape=[1,3,2,2])
c = b - a
d = tf.square(c)
e = tf.reduce_sum(d)

sess = tf.InteractiveSession()
print(a)
print(b)

print(sess.run(e))