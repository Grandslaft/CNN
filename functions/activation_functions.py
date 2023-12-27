import numpy as np

class Activation:
    def __init__(self, name, alpha=1):
        self.name = name
        if name == 'sigmoid':
            self.function = self.exponent
            self.derivative = self.exponent_der
        elif name == 'tanh':
            self.function = lambda x: np.tanh(x)
            self.derivative = lambda x: 1 - np.square(np.tanh(x))
        elif name == 'softmax':
            self.function = self.softmax
            self.derivative = self.softmax_der
        elif name == 'relu':
            self.function = lambda x: np.maximum(0, x)
            self.derivative = lambda x: x >= 0
        elif name == 'lrelu':
            self.function = lambda x: np.maximum(0.0001, x) 
            self.derivative = lambda x: np.where(x >= 0, 1, 0.0001)
        elif name == 'elu':
            self.function = lambda x: np.where(x >= 0, x, alpha*(np.exp(x) - 1))
            self.derivative = lambda x: np.where(x >= 0, 1, self.function(x) + alpha)
        
    def exponent(self, x):
        shift_x = x - np.max(x)
        return 1/(1 + np.exp(-shift_x))
    
    def exponent_der(self, x):
        exps = self.exponent(x)
        return exps * (1 - exps)
    
    def softmax(self, x):
        shift_x = x - np.max(x)
        exps = np.exp(shift_x)
        return exps/np.sum(exps)
    
    def softmax_der(self, x):
        array = self.softmax(x).reshape(-1,1)
        return np.diagflat(array) - np.dot(array, array.T)