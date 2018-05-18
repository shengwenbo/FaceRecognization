import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import utils
import numpy as np
from batcher import Batcher
    
class VGG16(object):

    def __init__(self, width, height, num_classes=68, batch_size=32, dropout_keep_prob=0.7, learning_rate=0.01, log_dir="./vgg16_log/"):
        self.width = width
        self.height = height
        self.batch_size=batch_size
        self.dropout_keep_prob=dropout_keep_prob
        self.learning_rate = learning_rate
        self.log_dir=log_dir
        self.num_classes=num_classes

    def build_vgg16(self, inputs, is_training=True):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(
                inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [self.width//32, self.height//32], padding='VALID', scope='fc6')
            net = slim.dropout(
                net, self.dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(
                net, self.dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.conv2d(
                net,
                self.num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope='fc8')
        return net

    def train(self, images, labels):
        train_log_dir = self.log_dir
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)

        with tf.Graph().as_default() as graph:
            # images = slim.ops.convert_to_tensor(images)
            # labels = slim.ops.convert_to_tensor(labels)

            # image_batch, label_batch = utils.get_batch_data(images, labels, batch_size=self.batch_size)

            inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.width, self.height, 1), name="inputs")
            ouputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 1, 1, self.num_classes), name="outputs")

            predictions = self.build_vgg16(inputs, is_training=True)
            # predictions,_ = nets.vgg.vgg_a(images, num_classes=68)
            # Specify the loss function:


            tf.losses.softmax_cross_entropy(predictions, ouputs)
            # slim.losses.softmax_cross_entropy(predictions, label_batch)

            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('losses/total_loss', total_loss)

            # Specify the optimization scheme:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

            tf.logging.set_verbosity(tf.logging.INFO)
            best_loss = None
            # Actually runs training.
            with tf.Session() as sess:
                saver = tf.train.Saver()
                sess.run(tf.initialize_all_variables())
                tf.train.start_queue_runners(sess)
                batcher = Batcher(images, labels, self.batch_size)
                epoch = 0
                turn = 0
                total_turn = 0
                while(True):
                    real_images,real_labels,finised = batcher.next_batch()
                    if finised:
                        epoch += 1
                        turn = 0
                    real_labels = sess.run(tf.one_hot(real_labels, self.num_classes))
                    real_labels = sess.run(tf.reshape(real_labels, [real_labels.shape[0], 1, 1, real_labels.shape[1]]))
                    feed_dict={
                        "inputs:0": real_images,
                        "outputs:0": real_labels
                    }
                    _,loss = sess.run([train_tensor,total_loss], feed_dict)
                    turn += 1
                    total_turn += 1
                    if turn % 100 == 0:
                        tf.logging.info("epch: %d\tturn: %d" % (epoch, turn))
                        tf.logging.info("total loss: %f" % loss)
                        if best_loss is None or loss < best_loss:
                            saver.save(sess, self.log_dir+"vgg16", global_step=total_turn,
                                       latest_filename='checkpoint_best')
                            best_loss = loss
                            tf.logging.info("model saved")
                            print("")
