# Fully connected layer network. Applying pathnorm to it

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from NeuralNet import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

class Gam(object):
    def __init__(self, _in, _out):
        self._in = _in
        self._out = _out
        

class Train(object):
    def __init__(self):
        self.FLAGS_hlayers = [3] # The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
        self.FLAGS_noutputs = 10
        self.FLAGS_lr = 0.05
        self.FLAGS_steps = 1
        self.FLAGS_batchsize = 10000
        self.FLAGS_lambda = 10.0**6
        self.FLAGS_eta = 0.1
        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS_noutputs])

    def path_scale(self, gvs, depth):
        gamma = [Gam(None, None)] * depth
        gamma[0] = Gam(tf.ones([1, gvs[0][1].get_shape()[0]], tf.float32), None)
        gamma[depth-1] = Gam(None, tf.ones([gvs[len(gvs)-2][1].get_shape()[1], 1] , tf.float32))
        for i in range(1, depth):
            gamma[i]._in = tf.matmul(gamma[i-1]._in, tf.abs(tf.square(gvs[(i-1)*2][1]))) + tf.abs(tf.square(gvs[(i-1)*2+1][1]))
            j = depth-i-1
            gamma[j]._out = tf.matmul(tf.abs(tf.square(gvs[2*j][1])), gamma[j+1]._out)
            print(i, j, depth)
        return gamma            
        
    def capGrads(self, grads_and_vars, name = 'CONSTRAINT_PATH_NORM'):
        if name == 'SGD':
            return grads_and_vars

        elif name == 'CONSTRAINT_PATH_NORM':
            depth = len(self.FLAGS_hlayers)+2
            gamma = self.path_scale(grads_and_vars, depth)
            capped_grads_and_vars = [None]*len(grads_and_vars)
            for j in range(1, depth):


                gamma_j = tf.matmul(gamma[j-1]._in, gamma[j]._out, True, True)
                bias_j = tf.transpose(gamma[j]._out)
                wb_j = tf.concat([gamma_j, bias_j], 0)

                '''
                gamma_j_root = tf.sqrt(wb_j)
                gamma_j_root_dezero = tf.cond(tf.equal(tf.reduce_sum(gamma_j_root), 0), lambda: tf.add(gamma_j_root, 1.0), lambda: gamma_j_root) #

                gradient = grads_and_vars[(j-1)*2][0]
                gradientTheta = grads_and_vars[(j-1)*2+1][0]
                gradientTheta = tf.reshape(gradientTheta, [1, gradientTheta.get_shape().as_list()[-1]])
                wb_grad = tf.concat([gradient, gradientTheta], axis = 0)

                norm_grad_j = tf.norm(wb_grad)
                normed_grad = wb_grad / norm_grad_j
                c = tf.div(normed_grad, gamma_j_root_dezero) # Maybe gamma = 0

                # get the path norm components from the biases
                temp_bias_path_norm = 0.0
                for k_j in range(j+1, depth):
                    bias_k_j = grads_and_vars[(k_j-1)*2+1][1]
                    bias_k_j = tf.reshape(bias_k_j, [1, bias_k_j.get_shape().as_list()[-1]])
                    temp_bias_path_norm = temp_bias_path_norm + tf.matmul(tf.square(bias_k_j), gamma[k_j]._out)
                
                lambda_j = self.FLAGS_lambda - temp_bias_path_norm

                s_j = tf.multiply(-tf.sqrt(lambda_j), c)

                nrows, ncols = s_j.get_shape().as_list()
                w_j = tf.slice(s_j, [0, 0], [nrows-1, ncols])
                b_j = tf.slice(s_j, [nrows-1, 0], [1, ncols])
   
                '''
                # Updating the weights and biases
                capped_grads_and_vars[(j-1)*2] = (grads_and_vars[(j-1)*2][0], grads_and_vars[(j-1)*2][1])
                capped_grads_and_vars[(j-1)*2+1] = (grads_and_vars[(j-1)*2+1][0], grads_and_vars[(j-1)*2+1][1])
                '''
                grads_and_vars[(j-1)*2] = list(grads_and_vars[(j-1)*2])
                grads_and_vars[(j-1)*2+1] = list(grads_and_vars[(j-1)*2+1])
                grads_and_vars[(j-1)*2][1] = gamma_j

                grads_and_vars[(j-1)*2][1] = (1-self.FLAGS_eta)*grads_and_vars[(j-1)*2][1]# + self.FLAGS_eta*w_j
                grads_and_vars[(j-1)*2+1][1] = (1-self.FLAGS_eta)*grads_and_vars[(j-1)*2+1][1]#+ self.FLAGS_eta*b_j
                weight_zeros = tf.cond(tf.equal(tf.reduce_sum(grads_and_vars[(j-1)*2][1]), 0), lambda: 10.0**(-7), lambda: 0.0)
                grads_and_vars[(j-1)*2][1] = grads_and_vars[(j-1)*2][1] # #  weight_zeros


                grads_and_vars[(j-1)*2] = tuple(grads_and_vars[(j-1)*2])
                grads_and_vars[(j-1)*2+1] = tuple(grads_and_vars[(j-1)*2+1])
                '''
                
            return capped_grads_and_vars, gamma
                

    def prepareTrainData(self):
        # Read training dataset here
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data, train_labels = mnist.train.images[:, :], mnist.train.labels[:, :] 
        print('train features', len(train_data), len(train_data[0]))
        print('train labels', len(train_labels))
        return train_data, train_labels
        
    def train(self):

        # Preparing the training data
        train_data, train_labels = self.prepareTrainData()
        
        # Build the train graph
        logits = inference(self.image_placeholder, self.FLAGS_noutputs, self.FLAGS_hlayers[:], reuse = False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits))
        opt = tf.train.GradientDescentOptimizer(learning_rate = self.FLAGS_lr)

        '''
        grads_and_vars is a list of tuples. Each tuple is a (gradient, variable) kind
        For eg., grads_and_vars = [(g_Wh1, Wh1), (g_bh1, bh1), (g_Wh2, Wh2), (g_bh2, bh2)]
        '''
        grads_and_vars = opt.compute_gradients(loss)
        capped_grads_and_vars, gamma = self.capGrads(grads_and_vars)
        optimizer = opt.apply_gradients(capped_grads_and_vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

        # Start Training
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        for i in range(self.FLAGS_steps):

            # generate the batches
            ind = np.random.randint(len(train_data), size = self.FLAGS_batchsize)
            data_batch, labels_batch = train_data[ind, :], train_labels[ind, :]
            feed_dict = {self.image_placeholder: data_batch, self.labels_placeholder: labels_batch}
            _, train_loss, train_acc, c, g = sess.run([optimizer, loss, accuracy, grads_and_vars, gamma], feed_dict)
            #print(train_loss, train_acc, c)
            print(len(g))
            
        

def main():

    train = Train()
    train.train()

if __name__ == '__main__':
    main()
