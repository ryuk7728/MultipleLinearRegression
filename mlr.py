import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x,y,m,w,b):

    y_pred=np.dot(x,w)+b
    cost=np.sum((y_pred-y)**2)/(2*m)
    return cost

def derW(x,y,m,n,w,b):
    
    y_pred=np.dot(x,w)+b
    derw=np.dot(x.T,y_pred-y)/m
    return derw

def derB(x,y,m,n,w,b):
    
        y_pred=np.dot(x,w)+b
        derb=np.sum(y_pred-y)/m
        return derb

def grad_desc(x,y,m,n,w,b,iter,alpha):
    
    lossx=[]
    loss=[]
    finloss=0
    for i in range(iter):
         lossx.append(i)
         loss.append(compute_cost(x,y,m,w,b))
         tmp_w=w-alpha*derW(x,y,m,n,w,b)
         tmp_b=b-alpha*derB(x,y,m,n,w,b)
         w=tmp_w
         b=tmp_b
         if(i==iter-1):
              finloss=compute_cost(x,y,m,w,b)
    
    plt.plot(lossx,loss,color='red')
    plt.show()
    return w,b,finloss



x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
xmax=np.max(x_train)
x_train=x_train/xmax
y_train=np.array([460,232,178])
m=np.shape(x_train)[0]
n=np.shape(x_train)[1]
w=np.zeros((n,))
b=0
alpha=1
iter=1000000
w,b,finloss=grad_desc(x_train,y_train,m,n,w,b,iter,alpha)



print(f"Weights obtained:{w}, Bias Obtained: {b}, Final Loss:{finloss}")
