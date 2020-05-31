import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(7)
def sigmoid(z):
	return 1/(1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def log_loss(y,y_hat):
	l= -y*np.log(y_hat)
	l-=(1-y)*np.log(1-y_hat)
	return l

def assign(i,j,label):
	if label==j:
		return 1
	elif label==i:
		return 0
	return -1


def predict(w,x):
	z= x.dot(w)
	h= sigmoid(z)
	if(h>=0.5): return 1
	return 0

epochs=500
lr=0.1

df= pd.read_excel('data4.xlsx',header=None)

data= np.array(df)
X= data[:,:-1]
X-=np.mean(X,axis=0)
X/=np.std(X,axis=0)

bias= np.ones((X.shape[0],1))
X= np.hstack((X[:,:-1],bias,X[:,-1:]))
y= data[:,-1]
y-=1
classes= len(list(set(y)))
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.6,shuffle=True)


weights=[]
losses=[]

for i in range(classes):
	row=[]
	for j in range(classes):
		row.append(np.random.randn(X.shape[1]))
	weights.append(row)

plot_labels=[]
for i in range(classes):
	for j in range(i+1,classes):
		w=weights[i][j]
		y_temp= [assign(i,j,label) for label in y_train]
		temp= [y>=0 for y in y_temp]
		y_temp= np.array(y_temp)
		y_temp= y_temp[temp]
		X_temp= X_train[temp]
		row=[]
		plot_labels.append("{0} vs {1}".format(i,j))
		for e in range(epochs):
			z= X_temp.dot(w)
			y_hat= sigmoid(z)
			loss= log_loss(y_temp,y_hat)
			epoch_loss= np.sum(loss)/y_temp.shape[0]
			diff= y_hat-y_temp
			loss_der= diff.T.dot(X_temp)/X_temp.shape[0]
			row.append(epoch_loss)
			print("Class {0} vs {1} classifier  Epoch: {2}  Loss: {3}".format(i,j,e,row[-1]))
			w-=lr*loss_der
			weights[i][j]=w
		losses.append(row)

for row in losses:
	plt.plot(row,label= plot_labels[losses.index(row)])
plt.title('Multiclass Logistic Regression | One v One Classifiers',loc='center')
plt.xlabel('No. of epochs')
plt.ylabel('Training loss')
plt.legend(loc='upper right')
plt.show()

correct=0
y_class= list(y_test)
y_dist= [y_class.count(y) for y in set(y_class)]
y_dist_correct=[0]*classes

for k in range(len(X_test)):
	x=X_test[k]
	votes= [0]*classes
	for i in range(classes):
		for j in range(i+1,classes):
			if predict(weights[i][j],x)==1: votes[j]+=1
			else: votes[i]+=1
	y_pred= votes.index(max(votes))
	print("Predicted:{0} Actual:{1}".format(y_pred,y_test[k]))
	if y_pred==y_test[k]:
		y_dist_correct[y_pred]+=1
		correct+=1


for i in range(classes):
	class_accuracy= y_dist_correct[i]*100/y_dist[i]
	print("The accuracy for class {0} is {1}%".format(i+1,round(class_accuracy)))

print("Out of {0} test cases, {1} were predicted correctly, so the overall accuracy of the model is {2}%".format(len(X_test),correct,correct*100/len(X_test)))