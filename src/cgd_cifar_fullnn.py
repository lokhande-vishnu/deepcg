"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
from norms import *
from cifarlib import data_helpers
import time
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CGD:
    def __init__(self, opt_type, grad_type, alpha, lamda):
        # Hyper-parameters
        self.model_name = 'models/cgd_cifar_fullnn.ckpt'
        self.IM_SIZE = 32
        self.IM_C_DIM = 3
        self.IM_PIXELS = 3072
        self.CLASSES = 10
        self.HIDDEN_UNITS = 4000
        self.opt_type = opt_type
        self.grad_type = grad_type
        # input
        self.x = tf.placeholder(tf.float32, shape=[None, self.IM_PIXELS])
        self.y_true = tf.placeholder(tf.int64, shape=[None])
        # network weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        initializer = tf.contrib.layers.xavier_initializer()
        self.W_1 = weight_variable([self.IM_PIXELS, self.HIDDEN_UNITS])
        self.W_2 = weight_variable([self.HIDDEN_UNITS, self.HIDDEN_UNITS])
        self.W_3 = weight_variable([self.HIDDEN_UNITS, self.CLASSES])
        self.b_1 = bias_variable([self.HIDDEN_UNITS])
        self.b_2 = bias_variable([self.HIDDEN_UNITS])
        self.b_3 = bias_variable([self.CLASSES])

        self.theta_w = [self.W_1, self.b_1, self.W_2, self.b_2, self.W_3, self.b_3]
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

    def nnet(self, x):
        h1 = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W_1) + self.b_2)
        y_pred = tf.matmul(h2, self.W_1) + self.b_3
        return y_pred

    def calculate_grad(self, grad_type, grads_and_vars):
        if grad_type == 1:  # sgd_update
            gvs = [(gv[0], gv[1]) for gv in grads_and_vars]
        elif grad_type == 2:  # norm_sgd_update
            gvs = [(get_cgd(gv[0], gv[1]), gv[1]) for gv in grads_and_vars]
        elif grad_type == 3:  # cgd_fn_update
            gvs = [(get_cgd(gv[0], gv[1], self.alpha, self.lamda, grad_type), gv[1]) for gv in grads_and_vars]
        elif grad_type == 4:  # cgd_nn
            g0_cgd_nn = grads_and_vars[0][0]
            w0_cgd_nn = grads_and_vars[0][1]
            s0_cgd_nn, st, st_r, M = get_cgd(g0_cgd_nn, w0_cgd_nn, self.alpha, self.lamda, grad_type)
            gvs = [(s0_cgd_nn, gv[1]) for gv in grads_and_vars]
        # end
        return gvs

    def train(self, data, n_iters, print_iters, batch_size):
        y_pred = self.nnet(self.x)
        # Calculate loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_true, logits=y_pred))
        # Compute the gradients for a list of variables.
        if self.grad_type > 0:
            grads_and_vars = self.opt.compute_gradients(loss)
            if self.grad_type == 4:
                gv_non_j = grads_and_vars[:2]
                update_for_non_j = [(cal_grad_set(gv, self.alpha, self.lamda, 4), gv[1]) for gv in gv_non_j]
                gv_j = grads_and_vars[2:]
                gv_j_grad = [gv[0] for gv in gv_j]
                gv_j_w = [gv[1] for gv in gv_j]
                sum_fb_norm = tf.reduce_sum([frobenius_norm(m_grad) for m_grad in gv_j_grad])
                update_for_j = [(get_cgd_with_st(gv[0] / sum_fb_norm, gv[1], self.alpha, self.lamda), gv[1]) for gv in gv_j]
                # from IPython import embed; embed()
                gvs = update_for_non_j + update_for_j
            else:
                gvs = self.calculate_grad(self.grad_type, grads_and_vars)
            #end
            solver = self.opt.apply_gradients(gvs, global_step=self.global_step)
        else:
            solver = self.opt.minimize(loss, var_list=self.theta_w)

        # Evaluatation
        correct_prediction = tf.equal(tf.argmax(y_pred,1), self.y_true)
        # Operation calculating the accuracy of the predictions
        accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        beginTime = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
            batches = data_helpers.gen_batch(list(zipped_data), batch_size, n_iters)
            # Training
            for i in range(n_iters):

                # Get next input data batch
                batch = next(batches)
                images_batch, labels_batch = zip(*batch)
                # from IPython import embed; embed()
                feed_dict_keepall = {
                  self.x: images_batch,
                  self.y_true: labels_batch,
                  self.keep_prob: 1.0
                }

                feed_dict_keepsome = {
                  self.x: images_batch,
                  self.y_true: labels_batch,
                  self.keep_prob: 0.5
                }
                train_accuracy, loss_ = sess.run(
                    [accuracy, loss], feed_dict_keepall)
                # w1_, w0_, s0_ = sess.run([w1_, w0_, s0_], feed_dict_keepall)
                # alpha_ = sess.run([self.alpha], feed_dict_keepall)
                if i % print_iters == 0:
                    print('Itr:',str(i) + '/' + str(n_iters), '\tTrain accuracy=', train_accuracy, '\tLoss =', loss_)
                    endTime = time.time()
                    print('Iteration time: {:5.2f}s'.format(endTime - beginTime))
                    beginTime = time.time()

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
    # mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    data_sets = data_helpers.load_data()
    n_iters = 30000
    fraction_print = 0.1
    print_iters = round(fraction_print*n_iters)
    batch_size = 100
    opt_type = 1
    grad_type = 3 # frobenius_norm
    alpha = 0.99990
    lamda = 4.0

    net = CGD(opt_type, grad_type, alpha, lamda)  # Adam - norm SGD
    net.train(data_sets, n_iters, print_iters, batch_size)
