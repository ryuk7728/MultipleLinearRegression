import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x,y,m,n,w,b):
    cost=0
    for i in range(m):
        cost += (1/(2*m))*np.sum(((x[i]*w+b)-y[i])**2)
        print(cost)
    return cost


x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460,232,178])
m=np.shape(x_train)[0]
n=np.shape(x_train)[1]
w=np.zeros((n,))
print(w)
b=0
print(compute_cost(x_train,y_train,m,n,w,b))
