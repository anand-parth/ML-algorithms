import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_excel('data.xlsx',header = None)

data = np.array(df)
data -= np.mean(data,axis=0)
data /= np.std(data,axis=0)
bias = np.ones((data.shape[0],1))

data= np.hstack((data[:,:-1],bias,data[:,-1:]))

X= data[:,:-1]
y= data[:,-1]


val = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)
w = np.dot(val,y)

print(w)