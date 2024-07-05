from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))

x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

print(x_train.shape)
print(x_test.shape)

model=Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))

model.add(MaxPool2D(2,2))

model.add(Flatten())

model.add(Dense(100,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10)
y_pred=model.predict(x_test)
a=y_pred.argmax(axis=1)
cm=confusion_matrix(a,y_test)
print(cm)
acc=accuracy_score(a,y_test)
print(f"Accuracy={round(acc*100,2)}%")
