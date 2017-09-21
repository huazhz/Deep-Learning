"""ResNet Train/Eval module.
"""
import time
import six
import sys

import eeg_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('mode', 'eval', 'train or eval.')

tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')

# ensemble1's log
tf.app.flags.DEFINE_string('log_root', '../log_1',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('train_dir', '../log_1/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '../log_1/eval',
                           'Directory to keep eval outputs.')

tf.app.flags.DEFINE_integer('eval_batch_count', 30,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')

tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def train(hps):
    """Training loop."""

    images, labels = eeg_input.build_input(hps.batch_size, FLAGS.mode)
    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    # SummarySaverHook: Saves summaries every N steps.
    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        # 如果指定这个了，就不用summary_writer了
        output_dir=FLAGS.train_dir,
        # 把构建模型时的summary和记录precision的summary合并
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)])
    )

    # Prints the given tensors every N local steps, every N seconds, or at end.
    # 每100次迭代log一次
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    # 自定义一个调整学习率的回调，其中的一些回调方法在每次run()时会被调用
    # https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/train/SessionRunHook
    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.01

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            # The run_values argument contains results of
            # requested ops/tensors by before_run().
            train_step = run_values.results
            if train_step < 3000:
                self._lrn_rate = 0.01
            elif train_step < 5000:
                self._lrn_rate = 0.001
            elif train_step < 8000:
                self._lrn_rate = 0.0001
            else:
                self._lrn_rate = 0.00001

    # https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        count = 0
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)
            if count % 100 == 0:
                tf.logging.info("count: %d", count)
            count += 1
            # 每1000次迭代，停下来测一次准确率
            if count == 1000:
                mon_sess.close()
                break


def evaluate(hps):
    """Eval loop."""
    images, labels = eeg_input.build_input(hps.batch_size, FLAGS.mode)
    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        # 加载模型
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(60)


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if FLAGS.mode == 'train':
        batch_size = 16
    elif FLAGS.mode == 'eval':
        batch_size = 32

    num_classes = 58

    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.01,
                               num_residual_units=4,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
