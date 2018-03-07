# Fully connected layer network. Applying pathnorm to it

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from NeuralNet import *
from cifar10_input import *
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import pandas as pd
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

'''
Code generating nan gradients when called compute_gradients. Happens after 5 iterations.

'''


class Train(object):
    def __init__(self):
        self.FLAGS_hlayers = [4000, 4000] # The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
        self.FLAGS_noutputs = 10
        self.FLAGS_nfeatures = 32*32*3
        self.FLAGS_steps = 20
        self.FLAGS_batchsize = 100
        self.FLAGS_lambda = 10.0**8
        self.FLAGS_eta = 10**(-6)
        self.FLAGS_padding_size = 0
        
        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, self.FLAGS_nfeatures])
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS_noutputs])

    def path_scale(self, gvs, depth):
        gamma = []
        for _ in range(depth):
            gamma.append([None, None])
        gamma[0] = [tf.ones([1, gvs[0][1].get_shape()[0]], tf.float32), None]
        gamma[depth-1] = [None, tf.ones([gvs[len(gvs)-2][1].get_shape()[1], 1] , tf.float32)]
        for i in range(1, depth-1):
            gamma[i][0] = tf.matmul(gamma[i-1][0], tf.square(tf.abs(gvs[(i-1)*2][1]))) + tf.square(tf.abs(gvs[(i-1)*2+1][1]))
            j = depth-i-1
            gamma[j][1] = tf.matmul(tf.square(tf.abs(gvs[2*j][1])), gamma[j+1][1])
            print(i, j, depth)
        return gamma            
    
    def capGrads(self, grads_and_vars, name = 'CONSTRAINT_PATH_NORM'):
        print(name)
        if name == 'SGD':
            return grads_and_vars

        elif name == 'CONSTRAINT_PATH_NORM':
            depth = len(self.FLAGS_hlayers)+2
            temp_grads_and_vars = [None]*len(grads_and_vars)            
            for i in range(len(grads_and_vars)):
                temp_grads_and_vars[i] = [grads_and_vars[i][0], grads_and_vars[i][1]]
                
            for j in range(1, depth):
                gamma = self.path_scale(temp_grads_and_vars, depth)
                gamma_j = tf.matmul(gamma[j-1][0], gamma[j][1], True, True)
                bias_j = tf.transpose(gamma[j][1])
                wb_j = tf.concat([gamma_j, bias_j], 0)
                
                gamma_j_root = tf.sqrt(wb_j)
                gamma_j_root_dezero = tf.add(tf.cast(tf.equal(gamma_j_root, 0), tf.float32), gamma_j_root)
                gradient = temp_grads_and_vars[(j-1)*2][0]
                gradientTheta = temp_grads_and_vars[(j-1)*2+1][0]
                gradientTheta = tf.reshape(gradientTheta, [1, gradientTheta.get_shape().as_list()[-1]])
                wb_grad = tf.concat([gradient, gradientTheta], axis = 0)
            
                norm_grad_j = tf.norm(wb_grad)
                normed_grad = wb_grad / (norm_grad_j)
                
                c = tf.div(normed_grad, gamma_j_root_dezero) # Maybe gamma = 0
                
                # get the path norm components from the biases
                temp_bias_path_norm = 0.0
                for k_j in range(j+1, depth):
                    bias_k_j = temp_grads_and_vars[(k_j-1)*2+1][1]
                    bias_k_j = tf.reshape(bias_k_j, [1, bias_k_j.get_shape().as_list()[-1]])
                    temp_bias_path_norm = temp_bias_path_norm + tf.matmul(tf.square(bias_k_j), gamma[k_j][1])
                print(temp_bias_path_norm)

                ########################## Making a change here
                lambda_j = tf.abs(self.FLAGS_lambda - temp_bias_path_norm)
        
                s_j = -tf.multiply(tf.sqrt(lambda_j), c)
                
                nrows, ncols = s_j.get_shape().as_list()
                w_j = tf.slice(s_j, [0, 0], [nrows-1, ncols])
                b_j = tf.slice(s_j, [nrows-1, 0], [1, ncols])
                b_j = tf.reshape(b_j, [gradientTheta.get_shape().as_list()[-1]])

                # Updating the weights and biases
                temp_grads_and_vars[(j-1)*2][1] = (1-self.FLAGS_eta)*temp_grads_and_vars[(j-1)*2][1] + self.FLAGS_eta*w_j
                temp_grads_and_vars[(j-1)*2+1][1] = (1-self.FLAGS_eta)*temp_grads_and_vars[(j-1)*2+1][1] + self.FLAGS_eta*b_j

                # for zero weight; don't do bias
                #temp_grads_and_vars[(j-1)*2][1] = tf.add(10**(-7)*tf.cast(tf.equal(temp_grads_and_vars[(j-1)*2][1], 0), tf.float32), temp_grads_and_vars[(j-1)*2][1])

                
            for i in range(len(grads_and_vars)):
                grads_and_vars[i] = ((grads_and_vars[i][1]-temp_grads_and_vars[i][1])/self.FLAGS_eta, grads_and_vars[i][1])
            
            return grads_and_vars
                

    def prepareTrainData(self):
        '''
        # MNIST dataset
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data, train_labels = mnist.train.images[:, :], mnist.train.labels[:, :] 
        print('train features', len(train_data), len(train_data[0]))
        print('train labels', len(train_labels), len(train_labels[0]))
        print(train_labels)
        #return train_data, train_labels
        '''
        
        # CIFAR10 dataset
        train_data, train_labels = prepare_train_data(padding_size = self.FLAGS_padding_size)
        train_data = np.reshape(train_data, [len(train_data), -1])
        train_labels = to_categorical(train_labels)
        print('train features', len(train_data), len(train_data[0]))
        print('train labels', len(train_labels), len(train_labels[0]))
        print(train_labels)
        return train_data, train_labels

        
    def train(self):

        # Preparing the training data
        train_data, train_labels = self.prepareTrainData()
        print('train_data', train_data)
        
        # Build the train graph
        logits,_ = inference(self.image_placeholder, self.FLAGS_noutputs, self.FLAGS_hlayers[:], reuse = False)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits))
        logits_softmax = tf.nn.softmax(logits)
        loss = tf.reduce_mean(-tf.reduce_sum(self.labels_placeholder * tf.log(logits_softmax), reduction_indices=[1]))

        opt = tf.train.GradientDescentOptimizer(learning_rate = self.FLAGS_eta)

        '''
        grads_and_vars is a list of tuples. Each tuple is a (gradient, variable) kind
        For eg., grads_and_vars = [(g_Wh1, Wh1), (g_bh1, bh1), (g_Wh2, Wh2), (g_bh2, bh2)]
        gradient index = 0
        variable/weight index = 1
        '''
        grads_and_vars = opt.compute_gradients(loss)
        capped_grads_and_vars= self.capGrads(grads_and_vars)

        optimizer = opt.apply_gradients(capped_grads_and_vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # Start Training
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        train_loss_list = []

        for i in range(self.FLAGS_steps):

            # generate the batches
            ind = np.random.randint(len(train_data), size = self.FLAGS_batchsize)
            data_batch, labels_batch = train_data[ind, :], train_labels[ind, :]
            #print(len(data_batch), len(data_batch[0]), len(data_batch[0][0]))
            #print(len(labels_batch), len(labels_batch[0]))

            feed_dict = {self.image_placeholder: data_batch, self.labels_placeholder: labels_batch}
            #c, g, a_, a1_, a2_ = sess.run([capped_grads_and_vars, grads_and_vars, a, a1, a2], feed_dict)
            logits_ = sess.run(logits, feed_dict)
            gvss = sess.run(grads_and_vars, feed_dict)
            cgvss = sess.run(capped_grads_and_vars, feed_dict)
            _, train_loss, train_acc = sess.run([optimizer, loss, accuracy], feed_dict)

            print('gvss', i)
            print(gvss)
            print('cgvss', i)
            print(cgvss)

            print(train_loss, train_acc)
            step_list.append(i)
            train_error_list.append(1-train_acc)
            train_loss_list.append(train_loss)


            '''
            print('printing g')
            for i in range(len(g)):
                print(len(g[i][0]), len(g[i][1]))
                print(g[i][0])
                print(g[i][1])
            
            print('printing c')
            for i in range(len(c)):
                print(len(c[i][0]), len(c[i][1]))
                print(c[i][0])
                print(c[i][1])
            '''
        df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list, 'train_loss':train_loss_list})
        df.to_csv('stat_pathnorm_trial.csv')

            
def main():
    
    train = Train()
    train.train()

if __name__ == '__main__':
    main()

'''
MNIST dataset: These parameters work well
        self.FLAGS_hlayers = [3] # The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
        self.FLAGS_noutputs = 10
        self.FLAGS_steps = 2000
        self.FLAGS_batchsize = 1000
        self.FLAGS_lambda = 10.0**6
        self.FLAGS_eta = 0.00001
        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS_noutputs])

CIFAR10 dataset: 
        self.FLAGS_hlayers = [4000, 4000] # The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
        self.FLAGS_noutputs = 10
        self.FLAGS_nfeatures = 36*36*3
        self.FLAGS_steps = 20000
        self.FLAGS_batchsize = 1000
        self.FLAGS_lambda = 10.0**6
        self.FLAGS_eta = 0
        self.FLAGS_padding_size = 2
        
        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, self.FLAGS_nfeatures])
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.FLAGS_noutputs])




'''
