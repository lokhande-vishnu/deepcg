# Fully connected layer network. Applying pathnorm to it

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from NeuralNet import *

def create(n):
    x = []
    for i in range(n):
        x.append(0)
    return x


class Train(object):
    def __init__(self):
        self.FLAGS_hlayers = [400, 400, 400] # The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
        self.FLAGS_noutputs = 10
        self.FLAGS_lr = 0.05
        self.FLAGS_steps = 1000
        self.image_placeholder = tf.placeholder(dtype=tf.float32,shape=[None, 784])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.FLAGS_noutputs])


    def capGrads(self, grads_and_vars, name = 'SGD'):
        if name == 'SGD':
            return grads_and_vars
    
        
    def train(self):

        # Read training dataset here
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data, train_labels = mnist.train.images[:1000, :], mnist.train.labels[:1000, :] # Picking 1000 images
        print('train features', len(train_data), len(train_data[0]))
        print('train labels', len(train_labels))
                

        # Build the train graph
        logits = inference(self.image_placeholder, self.FLAGS_noutputs, self.FLAGS_hlayers, reuse = False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=logits))
        opt = tf.train.GradientDescentOptimizer(learning_rate = self.FLAGS_lr)
        grads_and_vars = opt.compute_gradients(loss)
        capped_grads_and_vars = self.capGrads(grads_and_vars)
        optimizer = opt.apply_gradients(capped_grads_and_vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.label_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

        # Start Training
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        for i in range(self.FLAGS_steps):

            # generate the batches
        
            _, train_loss, train_acc = sess.run([optimizer, loss, accuracy], {self.image_placeholder: train_data, self.label_placeholder: train_labels})
            print(train_loss, train_acc)
        
    

def main():

    train = Train()
    train.train()

if __name__ == '__main__':
    main()
