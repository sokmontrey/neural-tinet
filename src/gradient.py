import numpy as np
import math
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

