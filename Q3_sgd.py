import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
import random

# Algorithm for stochastic gradient descent

# Let (x(i),y(i)) be the training example
# Cost(θ, (x(i),y(i))) = (1/2) Σ( hθ(x(i))  - y(i))**2.     Note : 1/2 and not 1/2m

# Jtrain(θ) = (1/m) Σ Cost(θ, (x(i),y(i)))

# Repeat 
# {
# 	For i=1 to m
# 	{
#     	θj = θj – (learning rate) * Σ( hθ(x(i))  - y(i))xj(i)
#     	For every j =0 …n
# 	} 
# }

df= pd.read_excel('data.xlsx',header = None)

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
		J = J + w[j]**2;
	return J
		
w = np.random.randn(X.shape[1])

iterations = 500
alpha = 0.01
lamda = 0.1

J_history = []
for _ in range(iterations):
	val, val2, val3 = [0,0,0]
	h = np.dot(X,w)
	for i in range(m):	
		val = (h[i] - y[i])*X[i][0]
		val2 = (h[i] - y[i])*X[i][1]
		val3 = (h[i] - y[i])*X[i][2]
		w[0] = w[0]*(1-alpha*lamda/m) - (alpha/m)*val
		w[1] = w[1]*(1-alpha*lamda/m) - (alpha/m)*val2
		w[2] = w[2]*(1-alpha*lamda/m) - (alpha/m)*val3
	J_history.append(computeCost(X,y,w))


print(w)
plt.plot(J_history)
plt.ylabel('Cost func')
plt.xlabel('No. of iterations')
plt.show()


# print(X[0].dot(w), y[0])
