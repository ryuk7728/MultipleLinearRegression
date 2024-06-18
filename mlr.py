import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x,y,m,w,b):
    cost=0
    for i in range(m):
        cost += ((np.dot(x[i],w)+b)-y[i])**2
    cost = (1/(2*m))*cost
    return cost

def derW(x,y,m,n,w,b):

    derw=np.zeros(n)
    for j in range(n):
        tderw=0
        for i in range(m):
            tderw+=((np.dot(x[i],w)+b)-y[i])*x[i][j]
        derw[j]=tderw
        
    derw/=m
    

    return derw

def derB(x,y,m,n,w,b):
    
        derb=0
        for i in range(m):
            derb+=((np.dot(x[i],w)+b)-y[i])
        derb/=m
        return derb

def grad_desc(x,y,m,n,w,b,iter,alpha,lossx,loss):
    
    for i in range(iter):
         lossx=np.append(lossx,i)
         loss=np.append(loss,compute_cost(x,y,m,w,b))
         tmp_w=w-alpha*derW(x,y,m,n,w,b)
         tmp_b=b-alpha*derB(x,y,m,n,w,b)
         w=tmp_w
         b=tmp_b
    
    plt.scatter(lossx,loss,color='red')
    plt.show()
    return w,b



x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460,232,178])
m=np.shape(x_train)[0]
n=np.shape(x_train)[1]
w=np.zeros((n,))
b=0
alpha=0.0000001
iter=100000
lossx=np.array([])
loss=np.array([])
w,b=grad_desc(x_train,y_train,m,n,w,b,iter,alpha,lossx,loss)


print(f"Weights obtained:{w}, Bias Obtained: {b}")