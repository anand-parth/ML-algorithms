import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 


df= pd.read_excel('data.xlsx',header=None)

data= np.array(df)
data-=np.mean(data,axis=0)
data/=np.std(data,axis=0)
bias= np.ones((data.shape[0],1))

data= np.hstack((data[:,:-1],bias,data[:,-1:]))

X= data[:,:-1]
y= data[:,-1]

m,n = [X.shape[0], X.shape[1]]

def computeCost(X, y, w):
	h = np.dot(X,w)
	J = 0
	for i in range(m):
		J = J + 1/(m*2) * ((h[i] - y[i])**2)
	for j in range(n):
		J = J + np.abs(w[j])
	return J

w = np.random.rand(X.shape[1])

iterations = 1000
alpha = 0.01

res = 0
J_history = []
lamda = 0.1

for _ in range(iterations):
	h=np.dot(X,w)
	val,val2,val3 = [0,0,0]
	for i in range(m):
		val = ((h[i]-y[i])*(X[i][0])) 
		val2 = ((h[i]-y[i])*(X[i][1]))
		val3 = ((h[i]-y[i])*(X[i][2]))
		w[0] = w[0]*(1-lamda*alpha*np.sign(w[0])/(m*2)) - ((alpha/m)*val)
		w[1] = w[1]*(1-lamda*alpha*np.sign(w[1])/(m*2)) - ((alpha/m)*val2)
		w[2] = w[2]*(1-lamda*alpha*np.sign(w[2])/(m*2)) - ((alpha/m)*val3)
	J_history.append(computeCost(X,y,w))

print(w)
plt.plot(J_history)
plt.ylabel('Cost func')
plt.xlabel('No. of iterations')
plt.show()


grid_size = 100
w0= np.linspace(-2,2,grid_size)
w1= np.linspace(-2,2,grid_size)

losses= np.zeros((grid_size,grid_size))

for i in range(grid_size):
	for j in range(grid_size):
		w= np.array([w0[i],w1[j],1])
		y_hat= X.dot(w)
		loss= 0.5*(y_hat-y)**2
		epoch_loss= np.sum(loss)/m
		losses[i][j]= epoch_loss

ax= plt.axes(projection='3d')
W0,W1= np.meshgrid(w0,w1)
# print(W0,W1)
ax.plot_surface(W0, W1, losses, rstride=1, cstride=1, 
			    cmap='viridis', edgecolor='none')

ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('Loss')
plt.title('Cost function')

plt.show()

