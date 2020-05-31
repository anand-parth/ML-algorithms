import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def predict(w,x):
	z= x.dot(w.T)
	y_hat= sigmoid(z)
	y_hat= list(y_hat)
	return y_hat.index(max(y_hat))

def log_loss(y,y_hat):
	l= -y*np.log(y_hat)
	l-=(1-y)*np.log(1-y_hat)
	return l

data = pd.read_excel('data4.xlsx',header=None)

X = data.iloc[:,:-1]
X = (X - X.mean(axis=0))/X.std(axis=0)
X= np.array(X)
y = data.iloc[:,-1]	
y = y - 1

classes= len(list(set(list(y))))
y=[[1 if i==c else 0 for i in range(classes)]for c in y]
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
m = len(X)

w = np.random.rand(classes,X.shape[1])
iterations = 300
alpha = 0.05
losses = []
for _ in range(iterations):
	z = X_train.dot(w.T)
	h = sigmoid(z)
	loss= log_loss(y_train,h)
	epoch_loss= np.sum(loss)/y_train.shape[0]
	# print(epoch_loss)
	losses.append(epoch_loss)
	gradient = np.dot((h - y_train).T, X_train) / y_train.shape[0]
	w -= alpha * gradient

plt.plot(losses)
plt.show()

correct=0
y_class= [list(y) for y in y_test]
y_class=[ y.index(1) for y in y_class]
y_dist= [y_class.count(y) for y in set(y_class)]
y_dist_correct=[0]*classes

print(X_test)

for i in range(len(X_test)):
	y_pred= predict(w,X_test[i])
	y_t=list(y_test[i])
	y_true= y_t.index(max(y_t))
	if y_pred==y_true:
		y_dist_correct[y_true]+=1
		correct+=1

for i in range(classes):
	class_accuracy= y_dist_correct[i]*100/y_dist[i]
	print("The accuracy for class {0} is {1}%".format(i+1,round(class_accuracy)))

print("Out of {1} test cases, {0} were predicted correctly, so the overall accuracy of the model is {2}%".format(correct,len(X_test),round(correct*100/len(X_test))))
