import numpy as np
import math
class Loss:
    def __init__(self): return None

    #good for regression
    def MeanSquareError(self, output, target, Derivative=False):
        if not Derivative:
            return np.sum( (output - target)**2 )/target.shape[0]
        else: 
            return 2.0 / target.shape[0] * (output - target)
    #good for classification
    def CrossEntropyLoss(self, output, target, Derivative=False):
        if not Derivative:
            return -(np.sum( target * np.log(output+1e-8) ))/target.shape[0]
        else:
            return (output - target) * target

