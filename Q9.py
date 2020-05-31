import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def loss(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))

def predict(w,x):
	z= x.dot(w.T)
	y_hat= sigmoid(z)
	y_hat= list(y_hat)
	return y_hat.index(max(y_hat))

df= pd.read_excel('data4.xlsx',header=None)

data= np.array(df)
X= data[:,:-1]
X-=np.mean(X,axis=0)
X/=np.std(X,axis=0)

bias= np.ones((X.shape[0],1))
X= np.hstack((X[:,:-1],bias,X[:,-1:]))
y_temp= data[:,-1]
y_temp-=1
classes= len(list(set(y_temp)))

y=[[1 if i==c else 0 for i in range(classes)]for c in y_temp]
y=np.array(y)

weights=[]
accuracies=[]
kf= KFold(n_splits=5,shuffle=True,random_state=2)
model=1

iterations = 100
alpha = 0.1
for train_index,test_index in kf.split(X):
	X_train= X[train_index]
	X_test= X[test_index]
	y_train= y[train_index]
	y_test= y[test_index]


	w= np.random.randn(classes,X.shape[1])
	losses=[]

	for _ in range(iterations):
		z = np.dot(X_train,w.T)
		h = sigmoid(z)
		l= loss(y_train,h)
		epoch_loss= np.sum(l)/y_train.shape[0]
		losses.append(epoch_loss)
		temp= h-y_train
		loss_der= temp.T.dot(X_train)/X_train.shape[0]
		w-=alpha*loss_der

	plt.plot(losses,label="Loss using Fold {0}".format(model))


	correct=0

	for i in range(len(X_test)):
		y_pred= predict(w,X_test[i])
		y_t=list(y_test[i])
		y_true= y_t.index(max(y_t))
		if y_pred==y_true:
			correct+=1

	accuracy=correct/len(X_test)*100
	print("Accuracy using Fold {0} is {1}%".format(model,round(accuracy,2)))
	accuracies.append(accuracy)
	weights.append(w)
	model+=1

accuracies= np.array(accuracies)
print("Average accuracy with K-Fold cross validation is {0} %".format(round(np.mean(accuracies),2)))

plt.legend(loc="upper right")
plt.show()


