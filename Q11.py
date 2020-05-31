import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_excel('data4.xlsx',header=None)
data = np.array(data)

X = data[:,:-1]
X = (X - X.mean(axis=0))/X.std(axis=0)
X= np.array(X)
y = data[:,-1]	
y = y - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)
c = len(list(set(list(y)))) #Find no. of c

y_test = list(y_test) #count operation only on lists
dist_y_test = [y_test.count(y) for y in list(set(y_test))]
print(dist_y_test)
y_train = list(y_train) 
dist_y_train = [y_train.count(y) for y in list(set(y_train))]
prob_y_train=[ dist/len(y_train) for dist in dist_y_train]
dist_y_correct= [0]*c
mean_tr=[]
covar_tr=[]

for i in range(c):
	X_temp=[]
	for j in range(X_train.shape[0]):
		if y_train[j]==i:
			X_temp.append(X_train[j])
	X_temp= np.array(X_temp)
	mean_tr.append(np.mean(X_temp,axis=0))
	covar_tr.append(np.cov(X_temp.T))

correct = 0

for j in range(X_test.shape[0]):
	likelihood= [prob_y_train[i]*np.exp(-0.5*(np.transpose(X_test[j]-mean_tr[i]).dot(np.linalg.inv(covar_tr[i])).dot(X_test[j]-mean_tr[i]))) for i in range(c)]
	y_pred= likelihood.index(max(likelihood))
	print("Predicted: {0}  Actual:{1}".format(y_pred,round(y_test[j])))
	if y_pred==y_test[j]:
		dist_y_correct[y_pred]+=1
		correct+=1

for i in range(c):
	class_accuracy= dist_y_correct[i]*100/dist_y_test[i]
	print("The accuracy for class {0} is {1}%".format(i+1,round(class_accuracy)))

print("Out of {1} test cases, {0} were predicted correctly, so the overall accuracy of the model is {2}%".format(correct,len(X_test),round(correct*100/len(X_test))))
