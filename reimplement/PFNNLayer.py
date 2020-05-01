import numpy as np
import tensorflow as tf

class Layer:
    # phase 是整列input的phase (n)
    def __init__(self, shape, rng, phase, name):

        """rng"""
        self.rng = rng

        """shape"""
        # param shape -- tuple not arr (phase, out, in)
        self.nslices        = shape[0]
        self.weight_shape   = shape
        self.bias_shape     = shape[:-1]

        """"alpha"""
        # (phase, out, in)
        self.alpha_W        = tf.Variable(self.initialize_W(), name = name+"_alpha")
        # (phase, out)
        self.alpha_b        = tf.Variable(self.initialize_b(), name = name+"_alpha")

        """phase params"""
        self.pindex_1, self.mu = self.get_control(phase)
        self.pindex_0            = (self.pindex_1-1) % self.nslices
        self.pindex_2            = (self.pindex_1+1) % self.nslices
        self.pindex_3            = (self.pindex_1+2) % self.nslices

        self.mu_b                = tf.expand_dims(self.mu, 1) # (n * 1)
        self.mu_W                = tf.expand_dims(self.mu_b, 1) # (n * 1 * 1)

        """weight and bias of this phase"""
        self.weight, self.bias = self.THETA()

    def initialize_W(self):
        bound = np.sqrt(6. / np.prod(self.weight_shape[-2:]))
        alpha_W = np.asarray(
            self.rng.uniform(low = -bound, high = bound, size = self.weight_shape), dtype = np.float32)
        return tf.convert_to_tensor(alpha_W, dtype = tf.float32)

    def initialize_b(self):
        return tf.zeros(self.bias_shape, tf.float32)

    def get_control(self, phase):
        pscale = self.nslices * phase # (n)
        mu = pscale % 1
        pindex_1 = tf.cast(pscale, 'int32') % self.nslices
        return pindex_1, mu

    def THETA(self):
        # weight of this phase
        y0_W = tf.nn.embedding_lookup(self.alpha_W, self.pindex_0)
        y1_W = tf.nn.embedding_lookup(self.alpha_W, self.pindex_1)
        y2_W = tf.nn.embedding_lookup(self.alpha_W, self.pindex_2)
        y3_W = tf.nn.embedding_lookup(self.alpha_W, self.pindex_3)
        W = cubic(y0_W, y1_W, y2_W, y3_W, self.mu_W)

        # bias of this phase
        y0_b = tf.nn.embedding_lookup(self.alpha_b, self.pindex_0)
        y1_b = tf.nn.embedding_lookup(self.alpha_b, self.pindex_1)
        y2_b = tf.nn.embedding_lookup(self.alpha_b, self.pindex_2)
        y3_b = tf.nn.embedding_lookup(self.alpha_b, self.pindex_3)
        b = cubic(y0_b, y1_b, y2_b, y3_b, self.mu_b)

        return W, b

def cubic(y0, y1, y2, y3, mu):
    return (
        (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu +
        (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu +
        (-0.5*y0+0.5*y2)*mu +
        (y1))

"""save given number of weights and bias for precomputation in real-time """
def save_network(alpha_W, alpha_b, nslices, cache_num, filename):
    print("save_network")
    # save resultant nn for 50 different control points
    for i in range(cache_num):
        """calculate the index and weights in phase function """
        pscale = nslices * (float(i)/cache_num)
        # weight
        mu = pscale % 1
        # index
        pindex_1 = int(pscale) % nslices
        pindex_0 = (pindex_1 - 1) % nslices
        pindex_2 = (pindex_1 + 1) % nslices
        pindex_3 = (pindex_1 + 2) % nslices

        # 3 layers of alpha passed in as tuple, len(alpha) = number of layers
        for j in range(len(alpha_W)):
            w = alpha_W[j]
            b = alpha_b[j]
            W = cubic(alpha_W[pindex_0],alpha_W[pindex_1],alpha_W[pindex_2],alpha_W[pindex_3],mu)
            B = cubic(alpha_b[pindex_0],alpha_b[pindex_1],alpha_b[pindex_2],alpha_b[pindex_3],mu)
            print("W B saved")
            W.tofile(filename + './nn/W%0i_%03i.bin' % (j,i))
            B.tofile(filename + './nn/b%0i_%03i.bin' % (j,i))

    # save phase function control for 3 layers
    for j in range(len(alpha_W)):
        print("control saved")
        alpha_W.tofile(filename + './control/alpha_W%0i.bin' % j)
        alpha_b.tofile(filename + './control/alpha_b%0i.bin' % j)
