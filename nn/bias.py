import numpy as np
import theano
import theano.tensor as T

from layer import Layer

class BiasLayer(Layer):

    def __init__(self, shape):

    	# share instance of all-zero initialization 
        self.b = theano.shared(name='b', value=np.zeros(shape, dtype=theano.config.floatX), borrow=True)
        		 # Return a SharedVariable Variable, initialized with a copy or reference of value.
        self.shape = shape
        self.params = [self.b]
    
    def __call__(self, input):
        b = T.addbroadcast(self.b, *[si for si,s in enumerate(self.shape) if s ==1])
        return input + b
        