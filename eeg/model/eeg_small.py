import tensorflow as tf

# 58个受试者
NUM_CLASS=58

IMAGE_WIDTH = 27
IMAGE_HEIGHT = 31
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH

def inference(images_placeholder, is_training,
              depth1, depth2, depth3, dense1_units, dense2_units,
              dropout_rate=0.5):
    """
    Build the eeg model

    :param images_placeholder: images placeholder of shape (None, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    :param is_training: training mode or test mode
    :param depth1: conv1 filter depth
    :param depth2: conv2 filter depth
    :param depth3: conv3 filter depth
    :param dense1_units: dense1 units number
    :param dense2_units: dense2 units number
    :return: softmax_linear: Output tensor with the computed logits.
    """
    training_mode = is_training is not None

    # layer1:bn-conv-relu(depth1)-pool
    with tf.name_scope('conv1'):
        print('images_placeholder.shape:', images_placeholder.shape)
        bn = tf.layers.batch_normalization(inputs=images_placeholder, training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=depth1,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv layer:', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    # layer2:bn-conv-relu(depth2)-pool
    with tf.name_scope('conv2'):
        bn = tf.layers.batch_normalization(inputs=pool, training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=depth2,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv layer:', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    # layer3:bn-conv-relu(depth3)-pool
    with tf.name_scope('conv3'):
        bn = tf.layers.batch_normalization(inputs=pool, training=training_mode)
        tf.summary.histogram('batch norm', bn)

        conv = tf.layers.conv2d(
            inputs=bn,
            filters=depth3,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        tf.summary.histogram('conv layer:', conv)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        tf.summary.histogram('pool', pool)

    with tf.name_scope('dense1'):
        pool_flat = tf.reshape(pool, [-1, 3 * 3 * 64])
        dense = tf.layers.dense(inputs=pool_flat, units=dense1_units, activation=tf.nn.relu)
        tf.summary.histogram('dense', dense)

    # dropout
    with tf.name_scope('dropout'):
        dropout = tf.layers.dropout(
            inputs=dense, rate=dropout_rate, training=training_mode)

    # dense2 58 output units
    with tf.name_scope('dense2'):
        logits = tf.layers.dense(inputs=dropout, units=58)
        tf.summary.histogram('dense2', dense)

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate, learning_rate_decay):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_step: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # decay the learning rate
    learning_rate = tf.train.exponential_decay(
        learning_rate, global_step, 200, learning_rate_decay, staircase=True)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_step = optimizer.minimize(loss, global_step=global_step)
    return train_step


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    labels = tf.to_int64(labels)
    correct = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # Return the number of true entries.
    return accuracy