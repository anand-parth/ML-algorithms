import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(100)

data = pd.read_excel('data3.xlsx',header=None)
data = np.array(data)

X = data[:,:-1]
y = data[:,-1]
y-=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


mean_tr=[]
covar_tr=[]

for i in range(2):
	X_temp=[]
	for j in range(X_train.shape[0]):
		if y_train[j]==i:
			X_temp.append(X_train[j])
	X_temp= np.array(X_temp)
	mean_tr.append(np.mean(X_temp,axis=0))
	covar_tr.append(np.cov(X_temp.T))



fp,fn,tp,tn = [0,0,0,0]
for j in range(X_test.shape[0]):
	likelihood= [0.5*np.exp(-0.5*(np.transpose(X_test[j]-mean_tr[i]).dot(np.linalg.inv(covar_tr[i])).dot(X_test[j]-mean_tr[i]))) for i in range(2)]
	y_pred= likelihood.index(max(likelihood))
	print("Predicted: {0}  Actual:{1}".format(y_pred,round(y_test[j])))
	if y_test[j]==1:
		if y_pred==1:
			tp+=1
		else:
			fn+=1
	else:
		if y_pred==0:
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

