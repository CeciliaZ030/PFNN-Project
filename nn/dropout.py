import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class DropoutLayer(Layer):
# dropping random units to prevent overfitting, but will increase training time

    # amount is the droping ratio, rng is a random number
    def __init__(self, amount=0.7, rng=np.random):
        self.amount = amount
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.params = []
        

    # taking the activation of the previous layer and manage to drop selected units
    def __call__(self, input):
        if self.amount < 1.0:
            return (input * self.theano_rng.binomial(
                size=input.shape, n=1, p=self.amount,
                dtype=theano.config.floatX)) / self.amount
        else:
            return input
