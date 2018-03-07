import tensorflow as tf
import numpy as np
import scipy.io as spio

allW= spio.loadmat('allW.mat', squeeze_me=True)['allW']
alltheta = spio.loadmat('alltheta.mat', squeeze_me=True)['alltheta']

print(allW)
print(alltheta)


def create_variables(name, shape, p, ch):
    if ch == 'w':
        w = tf.constant(allW[p-1], dtype=tf.float32)
        new_variables = tf.get_variable(name = name, initializer=w, dtype=tf.float32)
        return new_variables
    else:
        theta = tf.constant(alltheta[p-1], dtype=tf.float32)
        new_variables = tf.get_variable(name = name, initializer=theta, dtype=tf.float32)
        return new_variables
    #new_variables = tf.get_variable(name = name, shape = shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))#, initializer = tf.random_normal_initializer())


def inference(input, noutputs, hlayernum, reuse):
    layers = []
    Wbs = []
    layers.append(input)
    hlayernum.insert(0, input.get_shape().as_list()[-1])
    hlayernum.append(noutputs)
    with tf.variable_scope('fc', reuse=reuse):
        for i in range(len(hlayernum)):
            if i == 0:
                continue
            W = create_variables(name = 'weight%d' % i, shape = [hlayernum[i-1], hlayernum[i]], p=i, ch='w')
            b = create_variables(name = 'bias%d' % i, shape = [hlayernum[i]], p=i, ch='b')
            h = tf.nn.relu(tf.matmul(layers[i-1], W) + b )
            layers.append(h)
            Wbs.append(W)
            Wbs.append(b)
    return layers[-1], Wbs

    
