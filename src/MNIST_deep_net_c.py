"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
from norms import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CGD:
    def __init__(self, opt_type, grad_type, alpha, lamda):
        # Hyper-parameters
        self.model_name = 'models/cgd.ckpt'
        self.opt_type = opt_type
        self.grad_type = grad_type
        # input
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_true = tf.placeholder(tf.float32, shape=[None, 10])
        # network weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        initializer = tf.contrib.layers.xavier_initializer()
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.W_conv2 = weight_variable([5, 5, 32, 64])

        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_conv1 = bias_variable([32])
        self.b_conv2 = bias_variable([64])
        self.b_fc1 = bias_variable([1024])
        self.W = weight_variable([1024, 10])
        self.b = bias_variable([10])

        self.theta_w = [self.W_conv1, self.b_conv1, self.W_conv2, \
                    self.b_conv2, self.W_fc1, self.b_fc1, self.W, self.b]
        # dropout prob
        self.keep_prob = tf.placeholder(tf.float32)
        # training config
        self.global_step = tf.Variable(0, trainable=False)
        self.start_train = 0.99990  # Requires very high lambda for Cgd_Fn
        # self.alpha = tf.train.inverse_time_decay(
        #     self.start_train, self.global_step, k, 0)
        self.alpha = alpha
        # Lambda
        self.lamda = tf.placeholder_with_default(
            tf.constant(lamda), tf.constant(lamda).shape)
        if self.opt_type == 1:
            # Adam optimizer
            self.opt = tf.train.AdamOptimizer(1e-4)
        else:
            # Gradient Descent optimizer
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=self.alpha)

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

    def calculate_grad(self, type, grads_and_vars):
        if type == 1:  # sgd_update
            gvs = [(gv[0], gv[1]) for gv in grads_and_vars]
        elif type == 2:  # norm_sgd_update
            gvs = [(get_cgd(gv[0], gv[1]), gv[1]) for gv in grads_and_vars]
        elif type == 3:  # cgd_fn_update
            gvs = [(get_cgd(gv[0], gv[1], self.alpha, self.lamda, type), gv[1]) for gv in grads_and_vars]
        elif type == 4:  # cgd_nn
            g0_cgd_nn = grads_and_vars[0][0]
            w0_cgd_nn = grads_and_vars[0][1]
            s0_cgd_nn, st, st_r, M = get_cgd(g0_cgd_nn, w0_cgd_nn, self.alpha, self.lamda, type)
            gvs = [(s0_cgd_nn, gv[1]) for gv in grads_and_vars]
        # end
        return gvs

    def train(self, mnist, n_iters, print_iters, batch_size):
        y_pred = self.nnet(self.x)
        # Calculate loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_true, logits=y_pred))
        # Compute the gradients for a list of variables.
        if self.grad_type > 0:
            grads_and_vars = self.opt.compute_gradients(loss)
            # for J
            # for non J
            gv_non_j = grads_and_vars[:2]
            from IPython import embed; embed()
            update_for_non_j = [(cal_grad_set(gv, self.alpha, self.lamda, 4), gv[1]) for gv in gv_non_j]

            # update_for_non_j = [get_cgd(gv[0], gv[1], self.alpha, self.lamda, 4) for gv in grad_non_j]
            gv_j = grads_and_vars[2:]
            update_for_j = get_cgd(gv_j[0],gv_j[1], self.alpha, self.lamda, 3)

            gv = self.calculate_grad(self.grad_type, grads_and_vars)
            solver = self.opt.apply_gradients(gv, global_step=self.global_step)
        else:
            solver = self.opt.minimize(loss, var_list=self.theta_w)
        # Evaluatation
        correct_prediction = tf.equal(
            tf.argmax(y_pred, 1), tf.argmax(self.y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Training
            for i in range(n_iters):
                batch = mnist.train.next_batch(batch_size)
                feed_dict_keepall = {self.x: batch[0],
                                     self.y_true: batch[1], self.keep_prob: 1}
                feed_dict_keepsome = {
                    self.x: batch[0], self.y_true: batch[1], self.keep_prob: 0.5}
                train_accuracy, loss_ = sess.run(
                    [accuracy, loss], feed_dict_keepall)
                # w1_, w0_, s0_ = sess.run([w1_, w0_, s0_], feed_dict_keepall)
                # alpha_ = sess.run([self.alpha], feed_dict_keepall)
                if i % print_iters == 0:
                    print('Itr:',str(i) + '/' + str(n_iters), '\tTrain accuracy=', train_accuracy, '\toss =', loss_)

                    # print('Norm of iterates: w(t+1) =', LA.norm(w1_),
                    #       'w(t) =', LA.norm(w0_), 's(t) =', LA.norm(s0_))
                    # print('self.alpha', alpha_)
                sess.run(solver, feed_dict_keepsome)
            # Testing
            feed_dict_test = {self.x: mnist.test.images,
                              self.y_true: mnist.test.labels, self.keep_prob: 1}
            test_accuracy = sess.run(accuracy, feed_dict_test)
            print('test_accuracy', test_accuracy)
            # Saving
            saver = tf.train.Saver()
            saver.save(sess, self.model_name)


if __name__ == '__main__':
    # Inputs and outputs
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_iters = 20000
    fraction_print = 0.1
    print_iters = round(fraction_print*n_iters)
    batch_size = 50
    opt_type = 1
    grad_type = 3
    alpha = 0.99990
    lamda = 10.0
    net = CGD(opt_type, grad_type, alpha, lamda)  # Adam - norm SGD
    net.train(mnist, n_iters, print_iters, batch_size)
