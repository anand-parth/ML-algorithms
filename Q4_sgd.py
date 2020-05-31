import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(7)

cols=[0,1,'y']
train_size= 280
epochs=200
lr=0.0001

df= pd.read_excel('data.xlsx',names=cols)

data= np.array(df)
data-=np.mean(data,axis=0)
data/=np.std(data,axis=0)
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
losses_ind=[]
train_path=[]

for e in range(epochs):
	epoch_loss=0
	train_path.append(w)
	for i in range(train_size):
		y_hat= X_train[i].dot(w)
		loss= 0.5*(y_hat-y_train[i])**2
		losses_ind.append(loss)
		loss_der= (y_hat-y_train[i])*X_train[i]
		w-=loss_der*lr
		epoch_loss+=loss
	losses.append(epoch_loss/train_size)
	print("Epoch: {0} Loss: {1}".format(e,losses[-1]))

train_path=np.array(train_path)
losses_ind= np.array(losses_ind)

plt.plot(losses_ind)
plt.title('Individual instance Loss',loc='center')
plt.ylabel('Loss')
plt.show()
plt.close()
plt.plot(losses,c='b')
plt.title('Stochastic Gradeint Descent',loc='center')
plt.xlabel('No. of epochs')
plt.ylabel('Training loss')
plt.show()

w_sol= np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

print("Weights via Gradient Descent: {0}".format(w))
print("Weights via solving for minima using Vectorized Method: {0}".format(w_sol))
print("Deviation along each axis is {0}".format(np.abs(w-w_sol)))

