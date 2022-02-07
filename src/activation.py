import numpy as np
import math
class Activation:
    #stil need to improve reLU and numpy datatype as they always overflow when use RELU
    def __init__(self): return None

    def ReLU(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative:
            return np.maximum(0,x)
        else: return np.greater(x, 0) + 0
    def LeakyReLU(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative:
            return np.maximum(2.0/10.0*x, x)
        else: return (np.greater(x, 0)+0) + (np.less(x,0.001)+0)*2.0/10.0
    def Sigmoid(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative:
            return 1/(1+np.exp(-x))
        else: return x * (1 - x)
    def Tanh(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative:
            return np.tanh(x)
        else: return 1 - x**2
    def Softmax(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative:
            exp_sum = np.sum( np.exp(x) )
            return np.exp(x)/exp_sum
        else: return x * (1 - x)
    def Linear(self, x, Derivative=False):
        x = np.clip(x, -600, 600)
        if not Derivative: return x
        else: return 1
