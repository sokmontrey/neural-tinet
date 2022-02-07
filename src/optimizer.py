import numpy as np
import math
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

