import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def distance(c1,c2):
	return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2 + (c1[3]-c2[3])**2);


# K-means  Clustering
data = pd.read_excel('data2.xlsx',header = None)
data = np.array(data)

c1 = data[0]
c2 = data[1]
c1_temp = [0, 0, 0, 0]
c2_temp = [0, 0, 0, 0]
i = 0
p_error = distance(c1,c1_temp) + distance(c2,c2_temp)
exp_error = 0.000001

while(p_error > exp_error):
	cluster1 = []
	cluster2 = []
	i+=1
	for x in data:
		d1 = distance(c1,x)
		d2 = distance(c2,x)
		if(d1<d2):
			cluster1.append(x)
		else:
			cluster2.append(x)

	cluster1 = np.array(cluster1)
	cluster2 = np.array(cluster2)

	x1 = np.mean(cluster1[:,[0]])
	x2 = np.mean(cluster1[:,[1]])
	x3 = np.mean(cluster1[:,[2]])
	x4 = np.mean(cluster1[:,[3]])
	c1_temp = [x1, x2, x3, x4]
	y1 = np.mean(cluster2[:,[0]])
	y2 = np.mean(cluster2[:,[1]])
	y3 = np.mean(cluster2[:,[2]])
	y4 = np.mean(cluster2[:,[3]])
	c2_temp = [y1, y2, y3, y4]
	p_error = distance(c1,c1_temp) + distance(c2,c2_temp)
	c1 = c1_temp
	c2 = c2_temp

print(c1,c2,i)