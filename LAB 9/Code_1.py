from perceptron import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class perceptron(object):
    def __init__(self,rate=0.01,niter=10):
        self.rate=rate
        self.niter=niter
    def fit(self,x,y):
        """Fit training data
        x:training vectors,x.shape:[#samples,#features]
        y:Target values,y.shape:[#samples]
        """

        #weights
        self.weight=np.zeros(1+x.shape[1])

        #no. of misclassification
        self.errors=[] 
        for i in range(self.niter):
            err=0
            for xi,target in zip(x,y):
                delta_w=self.rate*(target-self.predict(xi))
                self.weight[1:]+=delta_w*xi
                self.weight[0]+=delta_w
                err+=int(delta_w!=0.0)
            self.errors.append(err)
        return self

    def net_input(self,x):
        """Calculate net input"""
        return np.dot(x,self.weight[1:])+self.weight[0]
    
    def predict(self,x):
        """Return class label after unit step"""
        return np.where(self.net_input(x)>=0.0,1,-1)

def plot_decision_regions(x,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min

df=pd.read_csv("IRIS.csv")
# print(df)
# print(df.tail())
df.iloc[145:150,0:5]
y=df.iloc[0:100,4].values
# print(y)
y=np.where(y=='Iris-setosa',-1,1)
# print(y)
x=df.iloc[0:100,[0,2]].values
# print(x)
plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


pn=perceptron(0.1,10)
pn.fit(x,y)
plt.plot(range(1,len(pn.errors)+1),pn.errors,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()