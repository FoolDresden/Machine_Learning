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
y  = data[:,[2]]


bias = np.expand_dims(np.ones([len(x1)]),axis = 1)
X = np.append(bias,x1,axis = 1)
X = np.append(X,x2,axis = 1)

Xt = np.dot(X.T,X)
Xinv = np.linalg.inv(Xt)
W = np.dot(np.dot(Xinv,X.T),y)

Y_test = np.dot(X,W)
print(W)
