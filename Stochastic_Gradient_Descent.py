import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_excel('data.xlsx',header = None)
data = np.array(data)

# Data Preprocesssing (Feature Scaling)
x1 = data[:,[0]]
x2 = data[:,[1]]
y  = data[:,2]
# print(y)
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
y_mean = np.mean(y)

x1_std = np.std(x1)
x2_std = np.std(x2)
y_std = np.std(y)

x1 = (x1 - x1_mean)/x1_std
x2 = (x2 - x2_mean)/x2_std
y = (y - y_mean)/y_std

bias = np.expand_dims(np.ones([len(x1)]),axis = 1)
X = np.append(bias,x1,axis = 1)
X = np.append(X,x2,axis = 1)

# Linear Regression
no_of_itr = []
w0=0
w1=0
w2=0
d_prec = 0.0001
no_of_itr = []
w1_ans = []
w2_ans = []
cost_func = []
a = 0
rep = 0
alpha = 0.0001
a_prec = 1
i1 = 0
while (a_prec > d_prec) and (rep<=5):
	for i in range(len(y)):
		r0 = ((w0 + w1*X[i][1] + w2*X[i][2]) - y[i]) * X[i][0]
		r1 = ((w0 + w1*X[i][1] + w2*X[i][2]) - y[i]) * X[i][1]
		r2 = ((w0 + w1*X[i][1] + w2*X[i][2]) - y[i]) * X[i][2]
		w0 = w0 - alpha*r0
		w1 = w1 - alpha*r1
		w2 = w2 - alpha*r2
		w1_ans.append(w1)
		w2_ans.append(w2)
		no_of_itr.append(i1);
		i1 = i1 + 1
		j=0
		for i in range(len(y)):
			j = j + (1/(2*len(y)) * (w0 + w1*X[i][1] + w2*X[i][2] - y[i]) ** 2)
		cost_func.append(j)
		a_prec = abs(j-a)
		a = j
	rep = rep + 5

	# no_of_itr.append(rep);
	
print(w0,w1,w2)
print(x1[0],x2[0],y[0])
# plt.plot(no_of_itr,cost_func);
plt.plot(no_of_itr,cost_func)
plt.title('Cost functions vs iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(111, projection='3d')

ax.plot(w1_ans, w2_ans, cost_func)
#ax.plot_trisurf(w1_ans,w2_ans,cost_func)
ax.set_title('Cost functions vs Weights')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost function')
plt.show()

print(w0,w1,w2)