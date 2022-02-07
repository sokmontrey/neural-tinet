import numpy as np
import math
class Model:
    def __init__(self, 
            shape, 
            activation_function,
            loss_function):

        self.shape = shape
        self.act_func = activation_function
        self.loss_func = loss_function

        self.layer = []
        for i in shape:
            self.layer.append( np.zeros(i)
                    .astype(np.longdouble) )
        self.layer = np.array(self.layer, dtype=object)

        self.weight = []
        self.bias = []
        for i in range(len(shape) - 1):
            self.weight.append( (np.random.random_sample(
                (shape[i], shape[i+1]) )*2 -1)
                .astype(np.longdouble) )

            self.bias.append((
                np.random.random_sample((shape[i+1]))*2 -1)
                .astype(np.longdouble))
        self.weight = np.array(self.weight, dtype=object)
        self.bias = np.array(self.bias, dtype=object)

    def print(self): #just a function to print the model's parameters in a better form
        for i in range(self.weight.shape[0]):
            print(self.weight[i], " ", self.bias[i], "\n")
        for i in self.layer:
            print(i)

    def forward(self, input): #(forward propagation)
        self.layer[0] = np.array(input, dtype=np.longdouble)
        self.layer[0] /= np.max(self.layer[0]) + 1e-08 #TODO
        last_index = 0
        for i in range(self.weight.shape[0]):
            last_index = i+1
            self.layer[i+1]= np.dot(self.layer[i], self.weight[i]) + self.bias[i]
            self.layer[i+1]= self.act_func[i](self.layer[i+1])
        return self.layer[last_index]

    def loss(self, input, target): #model built in loss function instead of using otim's loss function
        target = np.array(target, dtype=np.longdouble)
        return self.loss_func( self.forward(input), target, Derivative=False)

