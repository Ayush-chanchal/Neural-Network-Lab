import numpy as np
import matplotlib.pyplot as plt


def leaky_ReLU(x):
    data = [max(0.05*value, value) for value in x]
    return np.array(data, dtype=float)


def der_leaky_ReLU(x):
    data = [1 if value > 0 else 0.05 for value in x]
    return np.array(data, dtype=float)


x_data = np.linspace(-10, 10, 100)
y_data = leaky_ReLU(x_data)
dy_data = der_leaky_ReLU(x_data)


plt.plot(x_data, y_data, x_data, dy_data)
plt.title('leaky ReLU Activation Function & Derivative')
plt.legend(['leaky_ReLU', 'der_leaky_ReLU'])
plt.grid()
plt.show()
