"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
from norms import *

class CGD(object):
    def __init__(self, opt_type, grad_type, keep_prob):
        # Hyper-parameters
        self.opt_type = opt_type
        self.grad_type = grad_type
        self.keep_prob = keep_prob
        # network weights
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_true = tf.placeholder(tf.float32, shape=[None, 10])
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.W = weight_variable([1024, 10])
        self.b = bias_variable([10])
        # dropout prob
        self.keep_prob = tf.placeholder(tf.float32)
        # training config
        self.global_step = tf.Variable(0, trainable=False)
        self.start_train = 0.9990  # Requires very high lambda for Cgd_Fn
        k = 1
        self.alpha = tf.train.inverse_time_decay(
            self.start_train, self.global_step, k, 0)
        # Lambda
        self.lam1 = tf.placeholder_with_default(
            tf.constant(4.0), tf.constant(4.0).shape)
        if self.opt_type == 1:
            # Adam optimizer
            self.opt = tf.train.AdamOptimizer(1e-4)
        else:
            # Gradient Descent optimizer
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.alpha)

    def weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    # Convolution and max pooling definitions
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def nnet(self, x):
        # Re-shape image in the required format # self.x_image: ? x (28x28x1)
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # First Conv + Max-pool layer #h_conv1: ? x (28x28x32), #h_pool1 ? x (14x14x32)
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # Second Conv + Max-pool layer #h_conv2: ? x (14x14x64), #h_pool2 ? x (7x7x64)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # Final fully connected layer to get good features #h_pool2: ? x (7x7x64), #h_fc1: 1 x 1024
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        # Drop-out for the final layer
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # Final Read-out layer and prediction
        y_pred = tf.matmul(h_fc1_drop, self.W) + self.b
        return y_pred

    def grad_update(self, type, grads_and_vars):
        if type == 1:  # sgd_update
            gv = [(gv[0], gv[1]) for gv in grads_and_vars]
        elif type == 2:  # norm_sgd_update
            gv = [(Sgdnm(gv[0], gv[1]), gv[1]) for gv in grads_and_vars]
        elif type == 3:  # cgd_fn_update
            gv = [(Cgd_Fn(gv[0], gv[1]), gv[1]) for gv in grads_and_vars]
        elif type == 4:  # cgd_nn
            g0_cgd_nn = grads_and_vars[0][0]
            w0_cgd_nn = grads_and_vars[0][1]
            s0_cgd_nn, st, st_r, M = Cgd_Nn(g0_cgd_nn, w0_cgd_nn)
            gv = [(s0_cgd_nn, gv[1]) for gv in grads_and_vars]
        # end
        return gv

    def train(self, mnist, n_iters, print_iters, batch_size):
        y_pred = self.nnet(self.x)
        # Calculate loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_true, logits=y_pred))
        # Compute the gradients for a list of variables.
        grads_and_vars = self.opt.compute_gradients(loss)
        gv = self.grad_update(self.grad_type, grads_and_vars)
        optimizer_gv = self.opt.apply_gradients(gv, global_step=self.global_step)
        # Evaluatation
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Training
            for i in range(n_iters):
                batch = mnist.train.next_batch(batch_size)
                feed_dict_keepall = {x: batch[0],y_true: batch[1], keep_prob: 1}
                feed_dict_keepsome = {x: batch[0], y_true: batch[1], keep_prob: self.keep_prob}
                train_accuracy, loss_= sess.run([accuracy, loss], feed_dict_keepall)
                # w1_, w0_, s0_ = sess.run([w1_, w0_, s0_], feed_dict_keepall)
                # alpha_ = sess.run([self.alpha], feed_dict_keepall)
                if i % print_iters == 0:
                    print('train_accuracy=', train_accuracy,'self.loss value =', loss_)
                    # print('Norm of iterates: w(t+1) =', LA.norm(w1_),
                    #       'w(t) =', LA.norm(w0_), 's(t) =', LA.norm(s0_))
                    # print('self.alpha', alpha_)
                sess.run(optimizer_gv, feed_dict_keepsome)
            # Testing
            feed_dict_test = {x: mnist.test.images,
                              y_true: mnist.test.labels, keep_prob: 1}
            test_accuracy = sess.run(accuracy, feed_dict_test)
            print('test_accuracy', test_accuracy)


if __name == '__main__':
    # Inputs and outputs
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_iters = 20000
    print_iters = 1000
    batch_size = 50
    opt_type = 1
    grad_type = 1
    keep_prob = 0.5
    net = CGD(opt_type, grad_type, keep_prob)  # Adam - norm SGD
    net.train(mnist, n_iters, batch_size)
