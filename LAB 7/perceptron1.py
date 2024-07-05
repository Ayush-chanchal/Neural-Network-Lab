import numpy as np
from collections import Counter

class Perceptron:
    def __init__(self,weights,bias=1,learning_rate=0.3):
        self.weights=np.array(weights)
        self.bias=bias
        self.learning_rate=learning_rate

    @staticmethod
    def unit_step_function(x):
        if x<=0:
            return 0
        else:
            return 1
    
    def __call__(self,in_data):
        bias_array = np.array([self.bias],dtype=int)
        in_data=np.concatenate((in_data,bias_array))
        result=self.weights @ in_data
        return Perceptron.unit_step_function(result)
    
    def adjust(self,target_result,in_data):
        if type(in_data)!=np.ndarray:
            in_data=np.array(in_data)
        calculated_result=self(in_data)
        error=target_result-calculated_result
        if error!=0:
            bias_array = np.array([self.bias],dtype=int)
            in_data=np.concatenate(in_data,bias_array)
            correction=error*in_data*self.learning_rate
            self.weights+=correction
    
    def evaluate(self,data,labels):
        evaluation=Counter()
        for sample,label in zip(data,labels):
            result=self(sample)
            if result==label:
                evaluation["correct"]+=1
            else:
                evaluation["wrong"]+=1
        return evaluation