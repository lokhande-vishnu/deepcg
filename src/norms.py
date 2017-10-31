import tensorflow as tf
import numpy as np
# # to implement nuclear norm
def unfold_conv_layer(W, option=True):
    sizes = tf.shape(W)
    return tf.reshape(W, shape=[sizes[0] * sizes[1], sizes[2], sizes[3]])


def cal_grad_set(gv, alpha, lamda, grad_type):
    (G, W) = gv
    g0 = unfold_conv_layer(G)
    w0 = unfold_conv_layer(W)
    sizes = tf.shape(w0)
    s = tf.ones(shape=[sizes[0], 1, sizes[2]])
    k = tf.constant(0)
    def body(k, s):
        r_2 = get_cgd(g0[:, k, :], w0[:, k, :], alpha, lamda, grad_type)
        r_3 = tf.expand_dims(r_2, 1)
        s = tf.concat([s, r_3], axis=1)
        k = k + 1
        return k, s
    def condition(k,s):
        return k < sizes[1]

    _, s = tf.while_loop(cond=condition, body=body, loop_vars=[k, s], shape_invariants=[k.get_shape(), tf.TensorShape([None,  None, None])])

    s = s[:, 1:, :]
    # s = tf.transpose(s, perm=[1, 0, 2])
    g_new = tf.reshape(s, shape=tf.shape(W))
    return g_new


def get_cgd(grad, wt, alpha, lamda, grad_type):
    if grad_type == 3: # F
        st = grad / frobenius_norm(grad)
    elif grad_type == 4:
        st = top_singular_vector(grad)
    return ((1 - alpha) / alpha) * (wt + lamda * st)

def get_cgd_with_st(st, wt, alpha, lamda):
    return ((1 - alpha) / alpha) * (wt + lamda * st)

def Sgdnm(grad, wt):
    return grad  # (grad / tf_frobenius_norm(grad))

def frobenius_norm(M):
    return tf.reduce_sum(M ** 2) ** 0.5

def top_singular_vector(M):
    st, ut, vt = tf.svd(M, full_matrices=False)
    M_size = tf.shape(M)
    ut = tf.reshape(ut[:, 0], [M_size[0],1])
    vt = tf.reshape(vt[:, 0], [M_size[1],1])
    st = tf.matmul(ut,tf.transpose(vt))
    return st
#
# G = tf.placeholder(tf.float32, shape=[5, 5, 4, 32])
# W = tf.placeholder(tf.float32, shape=[5, 5, 4, 32])
# alpha = tf.placeholder(tf.float32)
# lamda = tf.placeholder(tf.float32)
# gtype = tf.placeholder(tf.int32)
# k_n, g_n = cal_grad_set(G, W, alpha, lamda, gtype)
#
# # resultx, resulti  = tf.while_loop(condition, body, [x,i], shape_invariants=[tf.TensorShape([None]),i.get_shape()])
#
# if __name__ == '__main__':
#     g = np.random.rand(5, 5, 4, 32)
#     w = np.random.rand(5, 5, 4, 32)
#     with tf.Session() as sess:
#         tf.initialize_all_variables().run()
#         sz = sess.run(g_n, feed_dict={G:g, W:w, alpha: 0.51, lamda: 10, gtype:4})
#         from IPython import embed; embed()
