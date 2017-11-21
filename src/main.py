import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
from norms import *
import os
import time
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cgd_combined import CGD

# DEFINE CONSTANTS
flags = tf.app.flags
flags.DEFINE_integer("n_iters", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002,
                   "Learning rate of for adam [0.0002]")
flags.DEFINE_float("print_iters", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108,
                     "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64,
                     "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA",
                    "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg",
                    "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples",
                    "Directory name to save the image samples [samples]")
flags.DEFINE_boolean(
    "train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean(
    "crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False,
                     "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

self.model_name = 'models/cgd.ckpt'
self.opt_type = opt_type
self.grad_type = grad_type

def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_iters = 3000
    fraction_print = 0.1
    print_iters = round(fraction_print * n_iters)
    batch_size = 100
    opt_type = 1
    grad_type = 4
    alpha = 0.99990
    lamda = 4.0
    net = CGD(opt_type, grad_type, alpha, lamda)  # Adam - norm SGD
    net.train(mnist, , , batch_size)

if __name__ == '__main__':
    tf.app.run()
