import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(17)

cols=[0,1,'y']
train_size= 280
epochs=200
lr=0.05

df= pd.read_excel('data.xlsx',names=cols)

data= np.array(df)
data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
bias= np.ones((data.shape[0],1))

data= np.hstack((data[:,:-1],bias,data[:,-1:]))

X= data[:,:-1]
y= data[:,-1]

X_train= X[:train_size]
X_test= X[train_size:]
y_train= y[:train_size]
y_test= y[train_size:]

w= np.random.randn(X.shape[1])
losses=[]

print(X_train,y_train)

for e in range(epochs):
	y_hat= X_train.dot(w)
	loss= 0.5*((y_hat-y_train)**2)
	epoch_loss= np.sum(loss)/train_size
	losses.append(epoch_loss)
	temp= y_hat-y_train
	temp= temp.reshape(train_size,1)
	loss_der= temp*X_train
	grad= loss_der.mean(axis=0)
	w-=lr*grad
	print("Epoch: {0} Loss: {1}".format(e,losses[-1]))

plt.plot(losses)
plt.title('Gradeint Descent',loc='center')
plt.xlabel('No. of epochs')
plt.ylabel('Training loss')
plt.show()

w_sol= np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

print("Weights via Gradient Descent: {0}".format(w))
print("Weights via solving for minima using Vectorization Method: {0}".format(w_sol))


