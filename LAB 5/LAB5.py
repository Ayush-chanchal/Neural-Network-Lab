import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_df=pd.read_csv("mnist_train.csv")
train_image=dict()
for id,row in train_df.iterrows():
    x=row[1:785].to_numpy()
    train_image[id]={'label': row[0],'data' : row[1:785].to_numpy() / np.sqrt(np.dot(x, x))}

def normalize(vec):
	return vec / np.sqrt(vec.dot(vec))

alpha= 0.20 
iterations = 100000                  
n_output   = 10                      
n_pixels   = 784                    
n_samples  = len(train_image)       
rng = np.random.default_rng()
w_chg_ls = list()
W = np.random.rand(n_output, n_pixels)
for x in range(n_output):
	W[x] = normalize(W[x])

def training():
	for t in  range(iterations):
		if t % 1000:
			w_chg_ls.append(np.copy(W))
		rand_i = rng.integers(n_samples)
		input_vec=train_image[rand_i]['data']
		win_index=np.argmax(np.dot(W,input_vec))
		W[win_index]+=alpha*(input_vec-W[win_index])
	return W
	    
weights = training()
def visualization():
	fig, ax = plt.subplots(nrows=10, ncols=5, figsize=(15, 15))
	for x in range(10):
		for w in range(5):
			ax[x,w].imshow(w_chg_ls[w*100][x].reshape((28,28)))
			ax[x,w].axis('off')
		plt.show()