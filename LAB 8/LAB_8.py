import numpy as np
import matplotlib.pyplot as plt

def function(a):
    if a>=0:
        return 1
    else:
        return 0
    
def NeuralNetwork(x,w,b):
    a=np.dot(x,w)+b
    y=function(a)
    return y

def Not_Gate(x):
    w=-1
    b=0.5
    return NeuralNetwork(x,w,b)

def And_Gate(x):
    w = np.array([1, 1])
    b= -1.5
    return NeuralNetwork(x, w, b)

def Or_Gate(x):
    w = np.array([1, 1])
    b= -0.5
    return NeuralNetwork(x, w, b)

def Nand_Gate(x):
    a=And_Gate(x)
    b=Not_Gate(a)
    return b

def Nor_Gate(x):
    a=Or_Gate(x)
    b=Not_Gate(a)
    return b

x1=0
x2=1

print("Not {}:".format(x1),Not_Gate(x1))
print("Not {}:".format(x2),Not_Gate(x2))
        
x1=np.array([0,0])
x2=np.array([0,1])
x3=np.array([1,0])
x4=np.array([1,1])

print("Or {}:".format(x1),Or_Gate(x1))
print("Or {}:".format(x2),Or_Gate(x2))
print("Or {}:".format(x3),Or_Gate(x3))
print("Or {}:".format(x4),Or_Gate(x4))

print("And {}:".format(x1),And_Gate(x1))
print("And {}:".format(x2),And_Gate(x2))
print("And {}:".format(x3),And_Gate(x3))
print("And {}:".format(x4),And_Gate(x4))

print("Nand {}:".format(x1),Nand_Gate(x1))
print("Nand {}:".format(x2),Nand_Gate(x2))
print("Nand {}:".format(x3),Nand_Gate(x3))
print("Nand {}:".format(x4),Nand_Gate(x4))

print("Nor {}:".format(x1),Nor_Gate(x1))
print("Nor {}:".format(x2),Nor_Gate(x2))
print("Nor {}:".format(x3),Nor_Gate(x3))
print("Nor {}:".format(x4),Nor_Gate(x4))
