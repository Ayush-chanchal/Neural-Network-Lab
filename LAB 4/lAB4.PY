import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dataset = pd.read_csv('IRIS.csv')
# print(dataset.head())

num_inputs = int(input("Enter the number of inputs: "))
num_samples = int(input("Enter the number of training samples: "))
train_set = [[int(input()) for x in range(num_inputs)]
             for y in range(num_samples)]

C = float(input("Enter the value of learning constant: "))


weights = [1, -1]


def sign_function(input_value):
    return (1 if input_value >= 0 else -1)


for iteration in range(len(train_set)):
    net_value = 0
    for i in range(len(weights)):
        net_value += weights[i] * train_set[iteration][i]

    signed_value = sign_function(net_value)
    delta = [C*signed_value*x for x in train_set[iteration]]
    for j in range(len(weights)):
        weights[j] = weights[j] + delta[j]

print(weights)
