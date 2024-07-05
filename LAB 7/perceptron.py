import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots()
xmin,xmax=-0.2,1.4
x=np.arange(xmin,xmax,0.1)
ax.set_xlim([xmin,xmax])
ax.set_ylim([-0.1,1.1])
m=-1
for m in np.arange(0,6,0.1):
    ax.plot(x,m*x)
ax.scatter(0,0,color="r")
ax.scatter(0,1,color="r")
ax.scatter(1,0,color="r")
ax.scatter(1,1,color="g")
plt.plot()
plt.show()