import numpy as np

def initialize_parameter(layers):
    parameters={}
    for i in range(1,len(layers)):
        parameters["W"+str(i)]=np.ones((layers[i-1],layers[i]))*0.1

    return parameters   


def forward_prop(X,parameters):
    A=X
    for i in range(1,len(parameters)+1):
        A_prev=A
        W1=parameters["W"+str(i)]
        # print(f"A{i}={A_prev}")
        # print(f"W{i}={W1}")
        A=np.dot(A,W1)
        # print(f"output={A}")
    return A,A_prev    

def update_parameter(X,layers,learning_rate,Y):
    parameters=initialize_parameter(layers)
    for epochs in range(20):
        for i in range(X.shape[0]):
            Y_hat,A=forward_prop(X[i],parameters)  
             
            for row1 in range(parameters["W2"].shape[0]):
                for col1 in range(parameters["W2"].shape[1]):
                    parameters["W2"][row1][col1]=parameters["W2"][row1][col1]+(learning_rate*(Y[i]-Y_hat)*2*[row1])
           

            for row in range(parameters["W1"].shape[0]):
                for col in range(parameters["W1"].shape[1]):
                    parameters["W1"][row][col]=parameters["W1"][row][col]+learning_rate*(Y[i]-Y_hat)*2*parameters["W2"][col][0]*X[i][row]

    return(parameters)        

X_train=np.array([[1,2,3],[4,5,6]])
Y_train=np.array([[1],[4]])
layers=[3,2,1]
weight_best=update_parameter(X_train,layers,0.01,Y_train)
x_test=np.array([[7,8,9],[10,11,12]])
y_test=np.array([[7],[10]])
for e in x_test:
    output,output_prev_layer=forward_prop(e,weight_best)
    print("prdicted output is",output)