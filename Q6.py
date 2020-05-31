import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_excel('data2.xlsx',header=None)
data= np.array(df)
train, test = train_test_split(data,train_size = 0.8, shuffle = True)

cols = data.shape[1]

c1 = train[np.random.randint(0,len(train)-1)]
c2 = train[np.random.randint(0,len(train)-1)]

def distance(X, pt):
    return np.sum(np.abs(X-pt))
def centroid(X):
    return np.mean(X, axis=0)

cA = []
cB = []
while(1):
    resA = []
    resB = []
    for i in range(len(train)):
        if(distance(train[i],c1) < distance(train[i], c2)):
            resA.append(train[i])
        else:
            resB.append(train[i])
    tempC = centroid(resA)
    if(distance(tempC, c1) <= 0.005):
        break
    cA = resA
    cB = resB
    c1 = centroid(cA)
    c2 = centroid(cB)
    # print(c1, c2)


Cluster= []
for t in train:
	if distance(t,c1)<distance(t,c2):
		Cluster.append(0)
	else:
		Cluster.append(1)
Cluster= np.array(Cluster)
f,ax= plt.subplots(len(train[0]),len(train[0])-1)
row=0
for i in range(len(train[0])):
	col=0
	for j in range(i+1,len(train[0])):
		ax[row,col].scatter(train[:,i],train[:,j],c=Cluster,s=10)
		col+=1
	row+=1
plt.suptitle("Training Clusters | k=2 | Each subplot compares 2 features")
plt.show()


Cluster= []
for t in test:
	if distance(t,c1)<distance(t,c2):
		Cluster.append(0)
	else:
		Cluster.append(1)
Cluster= np.array(Cluster)
f,ax= plt.subplots(len(test[0]),len(test[0])-1)
row=0
for i in range(len(test[0])):
	col=0
	for j in range(i+1,len(test[0])):
		ax[row,col].scatter(test[:,i],test[:,j],c=Cluster,s=10)
		col+=1
	row+=1
plt.suptitle("Testing Clusters | k=2 | Each subplot compares 2 features")
plt.show()