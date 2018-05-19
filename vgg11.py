import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import utils
import numpy as np
from batcher import Batcher
    
class VGG11(object):

    def __init__(self, width, height, num_classes=69, batch_size=32, dropout_keep_prob=0.7, learning_rate=0.001, log_dir="./vgg8_log/", mode="train"):
        self.width = width
        self.height = height
        self.batch_size=batch_size
        self.dropout_keep_prob=dropout_keep_prob
        self.learning_rate = learning_rate
        self.log_dir=log_dir
        self.num_classes=num_classes
        self.mode = mode

    def build_vgg11(self, inputs):
        is_training = self.mode == "train"
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(
                inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
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

    def eval(self, images, labels):
        with tf.get_default_graph().as_default():

            inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.width, self.height, 1), name="inputs")
            ouputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 1, 1, self.num_classes), name="outputs")

            predictions = self.build_vgg11(inputs)

            sess = tf.Session(config=utils.get_config())
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            ckpt_state = tf.train.get_checkpoint_state(self.log_dir, latest_filename="checkpoint")
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            # 不知道不够一个batch时怎么处理，所以干脆每个数据复制batch_size次

            logits = sess.run(predictions, feed_dict={"inputs:0":images})
            logits = logits.reshape(self.batch_size, self.num_classes)
            real_labels = np.argmax(logits, axis=1)

        # with tf.get_default_graph().as_default():
        #     sess = tf.Session(config=utils.get_config())
            # 计算评价指标
            labels_ph = tf.placeholder(shape=labels.shape, dtype=tf.int32, name="data_labels")
            real_labels_ph = tf.placeholder(shape=real_labels.shape, dtype=tf.int32, name="model_labels")
            accuracy,acc_op = tf.metrics.accuracy(labels_ph, real_labels_ph)
            sess.run(tf.local_variables_initializer())
            result = sess.run(acc_op, feed_dict={"data_labels:0":labels, "model_labels:0":real_labels})
            print("Accuracy: %f" % (result))

            # 打印错的
            for l1,l2 in zip(labels, real_labels):
                if l1 != l2:
                    print("expect:%d\tpredict:%d" % (l1, l2))




    def train(self, images, labels, load_model=True):
        train_log_dir = self.log_dir
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)

        with tf.Graph().as_default() as graph:

            # image_batch, label_batch = utils.get_batch_data(images, labels, batch_size=self.batch_size)

            inputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.width, self.height, 1), name="inputs")
            ouputs = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 1, 1, self.num_classes), name="outputs")

            predictions = self.build_vgg11(inputs)
            # Specify the loss function:


            tf.losses.softmax_cross_entropy(ouputs, predictions)

            total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('losses/total_loss', total_loss)

            # Specify the optimization scheme:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

            tf.logging.set_verbosity(tf.logging.INFO)
            best_loss = None
            # prepare
            saver = tf.train.Saver()
            sv = tf.train.Supervisor(logdir=self.log_dir,
                                     is_chief=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                     save_model_secs=60)  # checkpoint every 60 secs)
            summary_writer = sv.summary_writer
            tf.logging.info("Preparing or waiting for session...")
            sess_context_manager = sv.prepare_or_wait_for_session(config=utils.get_config())
            tf.logging.info("Created session.")
            # Actually runs training.
            with sess_context_manager as sess:
                batcher = Batcher(images, labels, self.batch_size)
                epoch = 0
                turn = 0
                total_turn = 0
                while(True):
                    real_images,real_labels,finised = batcher.next_batch()
                    if finised:
                        epoch += 1
                        turn = 0
                    real_labels = np.eye(self.num_classes)[real_labels]
                    real_labels = np.reshape(real_labels, [real_labels.shape[0], 1, 1, real_labels.shape[1]])
                    feed_dict={
                        "inputs:0": real_images,
                        "outputs:0": real_labels
                    }
                    _,loss,r = sess.run([train_tensor,total_loss,predictions], feed_dict)
                    turn += 1
                    total_turn += 1
                    if turn % 100 == 0:
                        tf.logging.info("epch: %d\tturn: %d/%d" % (epoch, turn,batcher.batch_count))
                        tf.logging.info("total loss: %f" % loss)
                        summary_writer.flush()
