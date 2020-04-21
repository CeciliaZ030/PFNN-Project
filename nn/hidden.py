import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class HiddenLayer(Layer):

    # weights_shape = tuple of (phase slices, input, output)

    def __init__(self, weights_shape, rng=np.random, gamma=0.01):
        
        # W initialzed randomly with certain W_bound

        W_bound = np.sqrt(6. / np.prod(weights_shape[-2:]))
        W = np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=weights_shape),
            dtype=theano.config.floatX)

        # create shared instance
        self.W = theano.shared(name='W', value=W, borrow=True)
        self.params = [self.W]
        self.gamma = gamma
        # learning rate
        
    def cost(self, input):
        return self.gamma * T.mean(abs(self.W))
        
    def __call__(self, input):
        return self.W.dot(input.T).T

        