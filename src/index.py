#example
import numpy as np
import math
import matplotlib.pyplot as plt

from model import Model
from activation import Activation
from loss import Loss
from gradient import Gradient
from optimizer import Optimizer

#create some activations function that will be used
act = Activation()
ReLU = act.ReLU
Tanh = act.Tanh

#create a loss function
MSE = Loss().MeanSquareError

#create a model with 4 layers(first one just for input)
#with 1, 4, 4, neurons and 1 neuron for output
#apply ReLU for hidden layers(not include input layer) Tanh for output
#and MSE (mean square error) for loss function
model = Model([1,4,4,1], [ReLU, ReLU, Tanh], MSE)

#create a optimier that use Gradient decent
optim = Optimizer(model, Gradient, MSE)
#tell optimizer object to use Momentum optimizer
optim.use_Momentum(learning_rate=0.03, momentum=0.1)

#do this step for entire dataset with multiple epochs to better train the model

#tell optimizer to fit a data of "1" and expected output of "5"
optim.fit([1], [5])
#optim.fit([1], [5], cal_loss=True) to automatically calculate loss
#and optim.loss to get loss

model.forward([1]) # to get prediction
