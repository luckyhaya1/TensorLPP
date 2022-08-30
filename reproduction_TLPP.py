import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits, load_iris
import spectral
import torch
import math
import scipy as sp
from scipy.io import loadmat
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from tensor_function import Patch,getU,kmode_product

image=loadmat('./Indian_pines.mat')['indian_pines_corrected']
label=loadmat('./Indian_pines_gt.mat')['indian_pines_gt']
num_Class = int(max(label.reshape(label.shape[0] * label.shape[1], 1)))
Height,Width,Band=image.shape
image=image.astype(float)
for band in range(Band): 
    image[:,:,band]=(image[:,:,band]-np.min(image[:,:,band]))/(np.max(image[:,:,band])-np.min(image[:,:,band]))
data=image
PATCH_SIZE=9

'''Divide the training set and test set and randomly select 10% as the training set'''
[Height, Width, Band] = data.shape
image_pad = np.zeros((Height + PATCH_SIZE -1, Width + PATCH_SIZE - 1, Band))
for band in range(Band):
    image_pad[:,:,band]=np.pad(data[:,:,band],int((PATCH_SIZE-1)/2),'symmetric')
data_patch_list = []
label_patch_list = []
for i in range(int((PATCH_SIZE - 1) / 2), data.shape[0] - 1 + int((PATCH_SIZE - 1) / 2)):
    for j in range(int((PATCH_SIZE - 1) / 2), data.shape[1] - 1 + int((PATCH_SIZE - 1) / 2)):
        if label[i-int((PATCH_SIZE - 1) / 2)][j-int((PATCH_SIZE - 1) / 2)]!=0:
            cut_patch = Patch(image_pad, i - int((PATCH_SIZE - 1) / 2), j - int((PATCH_SIZE - 1) / 2), PATCH_SIZE)  # 没问题
            data_patch_list.append(torch.from_numpy(cut_patch.transpose(1,2,0)))
            label_patch_list.append(label[i-int((PATCH_SIZE - 1) / 2)][j-int((PATCH_SIZE - 1) / 2)])
random_idx = np.random.choice(len(data_patch_list), int(0.1*len(data_patch_list)), replace=False)
X_train,train_label,X_test,test_label=[],[],[],[]
for m in random_idx:
    X_train.append(data_patch_list[m])
    train_label.append(label_patch_list[m])
idx_test = np.setdiff1d(range(len(data_patch_list)), random_idx)
for m in idx_test:
    X_test.append(data_patch_list[m])
    test_label.append(label_patch_list[m])

newshape=[1,1,20]
k_near=10
U1,U2,U3=getU(newshape,k_near,X_train,PATCH_SIZE,Band)
l=len(X_train)
x_train=[]
for i in range(l):
    x=kmode_product(X_train[i],U1,1)
    x=kmode_product(x,U2,2)
    x=kmode_product(x,U3,3)
    x_train.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))   
x_train= torch.tensor([item.detach().numpy() for item in x_train])
x_test=[]
l_t=len(X_test)
for i in range(l_t):
    x=kmode_product(X_test[i],U1,1)
    x=kmode_product(x,U2,2)
    x=kmode_product(x,U3,3)
    x_test.append(x.reshape(newshape[0]*newshape[1]*newshape[2]))
x_test= torch.tensor([item.detach().numpy() for item in x_test])

svc=svm.SVC(C=100,gamma=10,probability=True)
x_test= torch.tensor([item.detach().numpy() for item in x_test])
svc.fit(x_train,train_label)
acc=svc.score(x_test,test_label)
print(acc)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, train_label)
knn.score(x_train, train_label)
acc=knn.score(x_test, test_label)
print(acc)




