import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def loss(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))

def predict(X, w, threshold=0.5):
	h = sigmoid(np.dot(X,w))
	print(h)
	return h>=threshold

data = pd.read_excel('data3.xlsx',header=None)

X = data.iloc[:,:-1]
X = (X - X.mean(axis=0))/X.std(axis=0)
y = data.iloc[:,-1]	
y = y - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
m = len(X)

w = np.random.rand(X.shape[1])

iterations = 300
alpha = 0.05
losses = []
for _ in range(iterations):
	z = np.dot(X_train,w)
	h = sigmoid(z)
	l = loss(y_train,h)
	epoch_loss= np.sum(l)/y_train.shape[0]
	# print(epoch_loss)
	losses.append(epoch_loss)
	gradient = np.dot(X_train.T, (h - y_train)) / y_train.size
	w -= alpha * gradient
	
plt.plot(losses)
plt.title('Stochastic Gradient Descent',loc='center')
plt.xlabel('No. of epochs')
plt.ylabel('Training loss')
plt.show()

y_pred = predict(X_test,w)
print(y_pred)
fp,fn,tp,tn = [0,0,0,0]

y_test = np.array(y_test)
print(y_test)
for i in range(len(y_test)):
	if y_test[i]==1:
		if y_pred[i]==1:
			tp+=1
		else:
			fn+=1
	else:
		if y_pred[i]== 0:
			tn+=1
		else:
			fp+=1

print(tp,fp,tn,fn) 
sensitivity= tp/(tp+fn)
specificity= tn/(tn+fp)
accuracy= (tp+tn)/(tp+tn+fp+fn)

print("True Positives: ",tp)
print("False Positives: ",fp)
print("True Negatives: ",tn)
print("False Negatives: ",fn)
print("The sensitivity is {0}".format(sensitivity))
print("The specificity is {0}".format(specificity))
print("The overall accuracy is {0}".format(accuracy))
