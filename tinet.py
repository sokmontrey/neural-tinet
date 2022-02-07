import numpy as np
import math
import matplotlib.pyplot as plt

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

class Gradient: 
    def __init__(self, model):
        self.deri_act_func = model.act_func
        self.model = model

        self.weight_temp = []
        self.bias_temp = []
        for i in range(model.layer.shape[0]-1):
            self.weight_temp.append(np.array([]))
            self.bias_temp.append(np.array([]))

    def grad(self, G_Loss): #calculate gradient taking derivative of loss function 
        self.DY = G_Loss

        for i in range(self.model.layer.shape[0]-1, 0, -1):
            DA = self.DY * self.deri_act_func[i-1](
                    self.model.layer[i],Derivative=True )
            self.weight_temp[i-1] = np.matmul(
                self.model.layer[i-1].reshape(self.model.layer[i-1].shape[0], 1),
                DA.reshape(1, DA.shape[0]) )
            self.bias_temp[i-1] = DA
            if i-1:
                self.DY = np.dot(DA, self.model.weight[i-1].T)

        return np.array([self.weight_temp, self.bias_temp], dtype=object)

class Optimizer:
    def __init__(self, 
            model, 
            Gradient, 
            loss_function,
            batch_size=1):

        self.batch_size = batch_size
        self.gradient = Gradient(model)
        self.model = model
        self.loss_func = loss_function

        self.use_SGD()

    #call one of this method to tell optimizer class
    #what optimizer function to use
    def use_SGD(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.optim_func = self.SGD

    def use_Momentum(self,learning_rate=0.001, momentum=0.1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optim_func = self.Momentum

        self.old_grad_weight = np.array([], dtype=object)
        self.old_grad_bias = np.array([], dtype=object)

    def use_Adam(self, learning_rate=0.001, B1=0.9, B2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.B1 = B1
        self.B2 = B2
        self.epsilon = epsilon

        self.old_m_weight = np.array([], dtype=object)
        self.old_m_bias = np.array([], dtype=object)

        self.old_v_weight = np.array([], dtype=object)
        self.old_v_bias = np.array([], dtype=object)

    def SGD(self, grad_weight, grad_bias): #Socasitic gradient decent
        self.model.weight -= self.learning_rate * grad_weight
        self.model.bias -= self.learning_rate * grad_bias

    def Momentum(self, grad_weight, grad_bias): #momentum optimizer
        if not self.old_grad_weight.shape[0]:
            self.old_grad_weight = self.learning_rate * grad_weight
            self.old_grad_bias = self.learning_rate * grad_bias
        else:
            self.old_grad_weight = self.learning_rate * grad_weight +\
                self.momentum * self.old_grad_weight
            self.old_grad_bias = self.learning_rate * grad_bias +\
                self.momentum * self.old_grad_bias

        self.model.weight -= self.old_grad_weight
        self.model.bias -= self.old_grad_bias

    def Adam(self, grad_weight, grad_bias): #Adam optimizer. 
        if not self.old_m_weight.shape[0]:
            self.old_m_weight = (1 - self.B1) * grad_weight
            self.old_m_bias = (1 - self.B1) * grad_bias

            self.old_v_weight = (1 - self.B2) * grad_weight**2
            self.old_v_bias = (1 - self.B2) * grad_bias**2
        else:
            self.old_m_weight = self.B1 * self.old_m_weight +\
                    (1 - self.B1) * grad_weight
            self.old_m_bias = self.B1 * self.old_m_bias +\
                    (1 - self.B1) * grad_bias

            self.old_v_weight = self.B2 * self.old_v_weight +\
                    (1 - self.B2) * grad_weight**2
            self.old_v_bias = self.B2 * self.old_v_bias +\
                    (1 - self.B2) * grad_bias**2

        m_bar_weight = self.old_m_weight / (1 - self.B1)
        m_bar_bias = self.old_m_bias / (1 - self.B1)

        v_bar_weight = self.old_v_weight / (1 - self.B2)
        v_bar_bias = self.old_v_bias / (1 - self.B2)

        self.model.weight -= self.learning_rate * \
                m_bar_weight / \
                (np.square(v_bar_weight) + self.epsilon)
        self.model.bias -= self.learning_rate * \
                m_bar_bias / \
                (np.square(v_bar_bias) + self.epsilon)

    def fit(self, input, target, cal_loss=False): #calculate gradient with one input and one target
        predicted = self.model.forward(input)
        target = np.array(target, dtype=np.longdouble)
        G_Loss = self.loss_func(predicted, target, Derivative=True)
        if cal_loss:
            self.Loss = self.loss_func(predicted, target, Derivative=False)
        
        grad = self.gradient.grad(G_Loss)
        self.step(grad[0], grad[1])

    def batch_fit(self,  #calculate gradient with batche datas
            input, 
            target, 
            batch_size, 
            flatten=False, 
            cal_loss=False):

        grad = np.array([], dtype=object)
        target = np.array(target, dtype=np.longdouble)
        self.Loss = 0
        for i in range(batch_size):
            if flatten:
                predicted = self.model.forward(input[i].flatten())
            else: predicted = self.model.forward(input[i])
            G_Loss = self.loss_func(predicted, target[i], Derivative=True)
            if cal_loss:
                self.Loss += self.loss_func(predicted, 
                        target[i], 
                        Derivative=False)
            
            if not i:
                grad = self.gradient.grad(G_Loss)
            else: grad += self.gradient.grad(G_Loss)

        self.step(grad[0], grad[1])

    def batch_loss(self,input, target, batch_size, flatten=False): #calculate loss with batches data
        self.Loss = 0
        for i in range(batch_size):
            if flatten:
                predicted = self.model.forward(input[i].flatten())
            else: predicted = self.model.forward(input[i])
            self.Loss += self.loss_func(predicted, 
                    target[i], 
                    Derivative=False)
        return self.Loss

    def step(self, weight_grad, bias_grad): #update weight and bias
        self.optim_func(weight_grad, bias_grad)

#regression example

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



