import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_excel('data3.xlsx',header = None)
data = np.array(data)
np.random.shuffle(data)
# Data Preprocesssing (Feature Scaling)
x_temp = data[:,0:4]
y = data[0:59,-1] - 1 # y_train
y_test = data[60:99,-1] - 1
bias = np.expand_dims(np.ones([len(x_temp)]),axis = 1)
data = np.append(bias,x_temp,axis = 1)
X = data[0:59,0:5] # X_train
X_test = data[60:99,0:5]
# print(y_test)
# Linear Regression
w0=0
w1=0
w2=0
w3=0
w4=0
d_prec = 0.00001
cost_func = []
a = 0
rep = 0
alpha = 0.01
a_prec = 1
j = 0
while (a_prec > d_prec) and (rep<=1000):
	for i in range(len(y)):
		r0 = (1/(1 + np.exp(-1*(w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))) - y[i]) * X[i][0]
		r1 = (1/(1 + np.exp(-1*(w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))) - y[i]) * X[i][1]
		r2 = (1/(1 + np.exp(-1*(w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))) - y[i]) * X[i][2]
		r3 = (1/(1 + np.exp(-1*(w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))) - y[i]) * X[i][3]
		r4 = (1/(1 + np.exp(-1*(w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))) - y[i]) * X[i][4]
	w0 = w0 + alpha*r0
	w1 = w1 + alpha*r1
	w2 = w2 + alpha*r2
	w3 = w3 + alpha*r3
	w4 = w4 + alpha*r4
	for i in range(len(y)):
		hx = 1/(1 + np.exp(-1*((w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))))
		# print((w0 + w1*X[i][1] + w2*X[i][2] + w3*X[i][3] + w4*X[i][4]))
		j = j + y[i]*np.log(hx) + (1-y[i])*(np.log(1-hx))
	rep = rep + 1
	cost_func.append(j)
	a_prec = abs(j-a)
	a = j

c1=0
c2=0
c3=0
c4=0
for i in range(len(y_test)):
	if(w0 + w1*X_test[i][1] + w2*X_test[i][2] + w3*X_test[i][3] + w4*X_test[i][4] > 30):
		print('1',y_test[i])
		if(y_test[i] == 1.0):
			c1 += 1;
		else:
			c2 += 1;

	else:
		print('0',y_test[i])
		if(y_test[i] == 0.0):
			c3 += 1;
		else:
			c4 += 1;

print(c1,c2,c3,c4)




