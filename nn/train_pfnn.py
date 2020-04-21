import sys
import numpy as np
import theano
import theano.tensor as T
theano.config.allow_gc = True

sys.path.append('./nn')

from layer import Layer
from activation import ActivationLayer
from hidden import HiddenLayer
from bias import BiasLayer
from dropout import DropoutLayer
from trainer import AdamTrainer

""" Load Data """
//TODO


""" Phase Function Neural Network """

class PhaseFunctionedNetwork(Layer):
    
    # define input, output dimension and dropout ratio 
    def __init__(self, rng=rng, input_shape=1, output_shape=1, dropout=0.7):
        
        # number of phases    
        self.nslices = 4   
        # each layer has its own dropout
        self.dropout0 = DropoutLayer(dropout, rng=rng)
        self.dropout1 = DropoutLayer(dropout, rng=rng)
        self.dropout2 = DropoutLayer(dropout, rng=rng)
        # same activation function for all layers
        self.activation = ActivationLayer('ELU')
        
        # class HiddenLayer and BiasLayer are the actual params

        self.W0 = HiddenLayer((self.nslices, 512, input_shape-1), rng=rng, gamma=0.01)
        self.W1 = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.W2 = HiddenLayer((self.nslices, output_shape, 512), rng=rng, gamma=0.01)
    
        self.b0 = BiasLayer((self.nslices, 512))
        self.b1 = BiasLayer((self.nslices, 512))
        self.b2 = BiasLayer((self.nslices, output_shape))

        self.layers = [
            self.W0, self.W1, self.W2,
            self.b0, self.b1, self.b2]

        self.params = sum([layer.params for layer in self.layers], [])
        # for W -- self.params = [self.W] which W is randomly initialized nparray
        # for b -- self.params = [self.b] which b is nparray initialized as zeros
        
    def __call__(self, input):
        
        # pamount -- μ = (4p)mod 1, 
        # input[:,-1] is the last column which is the phase
        pscale = self.nslices * input[:,-1] 
        pamount = pscale % 1.0
        
        pindex_1 = T.cast(pscale, 'int32') % self.nslices #k1=4p%4
        pindex_0 = (pindex_1-1) % self.nslices #k0
        pindex_2 = (pindex_1+1) % self.nslices #k2
        pindex_3 = (pindex_1+2) % self.nslices #k3
        
        # dimshuffle - returns a view of this tensor with permuted dimensions.
        Wamount = pamount.dimshuffle(0, 'x', 'x')
        bamount = pamount.dimshuffle(0, 'x')
        
        # phase function with four control points
        # the weight of each layer is dynamically calculated by recombining different indices with phase 
        def cubic(y0, y1, y2, y3, mu):
            return (
                (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
                (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
                (-0.5*y0+0.5*y2)*mu +
                (y1))
        
        # 4 * 512 * 342
        W0 = cubic(self.W0.W[pindex_0], self.W0.W[pindex_1], self.W0.W[pindex_2], self.W0.W[pindex_3], Wamount)
        # 4 * 512 * 512
        W1 = cubic(self.W1.W[pindex_0], self.W1.W[pindex_1], self.W1.W[pindex_2], self.W1.W[pindex_3], Wamount)
        # 4 * 311 * 512
        W2 = cubic(self.W2.W[pindex_0], self.W2.W[pindex_1], self.W2.W[pindex_2], self.W2.W[pindex_3], Wamount)
        
        b0 = cubic(self.b0.b[pindex_0], self.b0.b[pindex_1], self.b0.b[pindex_2], self.b0.b[pindex_3], bamount)
        b1 = cubic(self.b1.b[pindex_0], self.b1.b[pindex_1], self.b1.b[pindex_2], self.b1.b[pindex_3], bamount)
        b2 = cubic(self.b2.b[pindex_0], self.b2.b[pindex_1], self.b2.b[pindex_2], self.b2.b[pindex_3], bamount)
        
        # Feed foward
        # input[:,:-1] values of the last phase
        H0 = input[:,:-1]
        # perform dot product with weight and input after dropping out some units. dropout(H)
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)
        H2 = self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1)
        H3 =                 T.batched_dot(W2, self.dropout2(H2)) + b2
        
        return H3
        
    def cost(self, input):
        input = input[:,:-1]
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)
    
    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))
        
    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))






