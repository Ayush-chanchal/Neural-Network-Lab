import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=load_iris()
x=data.data
y=data.target

y=pd.get_dummies(y).values
print(y[:3])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=4)
learning_rate=0.1
iterations=5000
n=y_train.size
input_size=4
hidden_size=2

output_size=3
results=pd.DataFrame(columns=["mse","accuracy"])

np.random.seed(10)
w1=np.random.normal(scale=0.5,size=(input_size,hidden_size))

w2=np.random.normal(scale=0.5,size=(hidden_size,output_size))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mean_squared_error(y_pred,y_true):
    return ((y_pred-y_true)**2).sum()/(2*y_pred.size)

def accuracy(y_pred,y_true):
    acc=y_pred.argmax(axis=1)==y_true.argmax(axis=1)
    return acc.mean()


for itr in range(iterations):
    z1=np.dot(x_train,w1)
    a1=sigmoid(z1)
    z2=np.dot(a1,w2)
    a2=sigmoid(z2)
    mse=mean_squared_error(a2,y_train)
    acc=accuracy(a2,y_train)
    results=results._append({"mse":mse,"accuracy":acc},ignore_index=True)
    e1=a2-y_train
    dw1=e1*a2*(1-a2)
    e2=np.dot(dw1,w2.T)
    dw2=e2*a1*(1-a1)

    w2_update=np.dot(a1.T,dw1)/n
    w1_update=np.dot(x_train.T,dw2)/n
    w2=w2-learning_rate*w2_update
    w1=w1-learning_rate*w1_update
results.mse.plot(title="Mean Squared Error")
plt.show()
results.accuracy.plot(title="Accuracy")
plt.show()

# Testing

z1=np.dot(x_test,w1)
a1=sigmoid(z1)

z2=np.dot(a1,w2)
a2=sigmoid(z2)

acc=accuracy(a2,y_test)
print("Accuracy:{}".format(acc))
