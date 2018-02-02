import tensorflow as tf
import numpy as np

def create_variables(name, shape):
    new_variables = tf.get_variable(name = name, shape = shape)
    return new_variables

def inference(input, noutputs, hlayernum, reuse):
    layers = []
    layers.append(input)
    hlayernum.insert(0, input.get_shape().as_list()[-1])
    hlayernum.append(noutputs)
    with tf.variable_scope('fc', reuse=reuse):
        for i in range(len(hlayernum)):
            if i == 0:
                continue
            W = create_variables(name = 'weight%d' % i, shape = [hlayernum[i-1], hlayernum[i]])
            b = create_variables(name = 'bias%d' % i, shape = [hlayernum[i]])
            h = tf.matmul(layers[i-1], W) + b 
            layers.append(h)
    return layers[-1]

    
