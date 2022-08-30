import numpy as np
import spectral
import torch
import math
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier

def Patch(data,H,W,PATCH_SIZE):
    transpose_array=np.transpose(data,(2,0,1)) #(C,H,W)
    height_slice=slice(H,H+PATCH_SIZE)
    width_slice=slice(W,W+PATCH_SIZE)
    patch=transpose_array[:,height_slice,width_slice]
    return np.array(patch)

def dist(x,k,t):
    n=len(x)
    s=np.zeros((n,n))
    w=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            s[i,j]=np.exp(-(math.pow(np.linalg.norm(x[j]-x[i]),2))/t)
    '''knn'''
    s=s+s.T
    for i in range(n):
        s[i,i]=s[i,i]/2
        index_=np.argsort(s[i])[-(k):]
        w[i,index_]=s[i,index_]
        w[index_,i]=s[index_,i]
        
    '''D'''
    for i in range(n):
        d[i,i]=sum(w[i,:])
    return w,d

def fold(matrix, mode, shape):
    """ Fold a 2D array into a N-dimensional array."""
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = torch.from_numpy(np.moveaxis(np.reshape(matrix, full_shape), 0, mode))
    return tensor

def unfold(tensor,mode):
    """ Unfolds N-dimensional array into a 2D array."""
    t2=tensor.transpose(mode,0)
    matrix = t2.reshape(t2.shape[0], -1)
    return matrix

def kmode_product(tensor,matrix,mode):
    """ Mode-n product of a N-dimensional array with a matrix."""
    ori_shape=list(tensor.shape)
    new_shape=ori_shape
    new_shape[mode-1]=matrix.shape[0]
    result=fold(np.dot(matrix,unfold(tensor,mode-1)),mode-1,tuple(new_shape))
    return result
def getyi_yj(Y,W):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        for j in range(i+1,l):
            if W[i][j]!=0:
                re=re+np.dot((Y[i]-Y[j]),(Y[i]-Y[j]).T)*W[i][j]*2
    return re

def getyy(Y,D):
    l=len(Y)
    re=np.zeros(np.dot(Y[0],Y[0].T).shape)
    for i in range(l):
        re=re+np.dot(Y[i],Y[i].T)*D[i][i]
    return re
            
def getvalvec(left,right,n_dims):
    eig_val, eig_vec = sp.linalg.eig(left,right)#np.linalg.pinv(right),left))    
    sort_index_ = np.argsort(np.abs(eig_val))
    eig_val = eig_val[sort_index_]
    print("eig_val:", eig_val[:1])
    j = 0
    while eig_val[j] < 1e-6:
        j+=1
    print("j: ", j)
    sort_index_ = sort_index_[j:j+n_dims]
    eig_val_picked = eig_val[j:j+n_dims]
    print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_] 
    return eig_vec_picked

def getU(newshape,k_near,X_train,P,Band):
    '''TLPP'''   
    w,d=dist(X_train,k_near,2)
    U1,U2,U3=np.eye(newshape[0],P),np.eye(newshape[1],P),np.eye(newshape[2],Band)
##    U=[0,U1,U2,U3]
##    U=[0.0,U1.astype(np.float64),U2.astype(np.float64),U3.astype(np.float64)]
    U1=U1.astype(np.float64)
    U2=U2.astype(np.float64)
    U3=U3.astype(np.float64)
    t_max=5
    l=len(X_train)
    for t in range(t_max):
        y1,y2,y3=[],[],[]
        for i in range(l):
            y=kmode_product(X_train[i],U2,2)
            y=kmode_product(y,U3,3)
            y1.append(unfold(y,0))
        left=getyi_yj(y1,w)  #(9,9)
        right=getyy(y1,d)
        newu1=getvalvec(left,right,newshape[0])
        print(newu1.dtype)
        
        lie=newu1.shape[1]
        for i in range(lie):
            newu1[:,i]=newu1[:,i]/np.linalg.norm(newu1[:,i],2)
        U1=newu1.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U3,3)
            y2.append(unfold(y,1))
        left=getyi_yj(y2,w)  #(9,9)
        right=getyy(y2,d)
        newu2=getvalvec(left,right,newshape[1])
        lie=newu2.shape[1]
        for i in range(lie):
            newu2[:,i]=newu2[:,i]/np.linalg.norm(newu2[:,i],2)
        U2=newu2.real.T
        
        for i in range(l):
            y=kmode_product(X_train[i],U1,1)
            y=kmode_product(y,U2,2)
            y3.append(unfold(y,2))
        left=getyi_yj(y3,w)  
        right=getyy(y3,d)
        newu3=getvalvec(left,right,newshape[2])
        lie=newu3.shape[1]
        for i in range(lie):
            newu3[:,i]=newu3[:,i]/np.linalg.norm(newu3[:,i],2)
        U3=newu3.real.T
    return U1,U2,U3
