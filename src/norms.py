# ### Norms
def tf_frobenius_norm(M):
    return tf.reduce_sum(M ** 2) ** 0.5

# # to implement nuclear norm
def tf_nuclear_norm(M):
    st, ut, vt = tf.svd(M,  full_matrices = False)
    #st2 = tf.diag(st)
    #st_r = tf.matmul(ut, tf.matmul(st2, tf.transpose(vt)))
    #print('vish', ut.shape, st2.shape, tf.transpose(vt).shape, st_r.shape)

    uk = tf.reshape(ut[:, 0], [10, 1])
    vk = tf.reshape(vt[:, 0], [1, 784])
    sk = tf.matmul(uk, vk)
    #sk = st[0] * sk
    #print(st.shape, ut.shape, vt.shape)
    #print('before', type(sk), sk.shape)
    return sk, _, _, _

def Sgdnm(grad, wt):
    return grad #(grad / tf_frobenius_norm(grad))

def Cgd_Fn(grad, wt):
    return ((1 - alpha ) / alpha) * (wt + lam1 * grad / tf_frobenius_norm(grad))

def Cgd_Nn(grad, wt):
    nn, st, st_r, M = tf_nuclear_norm(grad)
    return ((1 - alpha ) / alpha) * (wt - lam2 * nn), st, st_r, M
