#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import gzip
import imageio


# In[2]:


def enlarge(image,n):
    size=image.shape
    image2=np.zeros((n*size[0],n*size[1]))
    for i in range(n*size[0]):
        for j in range(n*size[1]):
            i2=int(i/n)
            j2=int(j/n)
            image2[i,j]=image[i2,j2]
    return image2


def sum_pooling(image,y,x):
    image_size=image.shape
    ymax=int(image_size[0]/y)
    xmax=int(image_size[1]/x)
    if (np.round(image_size[0]/y))!=(image_size[0]/y):
        print('please padding ',(image_size[0]-np.round(image_size[0]/y))*y,' in y direction')
        return 0
    if (np.round(image_size[1]/x))!=(image_size[1]/x):
        print('please padding ',(image_size[1]-np.round(image_size[1]/x))*x,' in x direction')
        return 0
    result=np.zeros((ymax,xmax))*1j
    for i in range(ymax):
        for j in range(xmax):
            buff=image[(i*y):((i+1)*y),(j*x):((j+1)*x)]
            result[i,j]=np.sum(buff)
    return result

def FFT(Figure,m):
    size=Figure.shape[1]
    size_mean=(size-1)/2
    m_mean=(m-1)/2
    Frequency=np.zeros((m,m))*1j
    N=size*m
    for i in range(m):
        for j in range(m):
            for i2 in range(size):
                for j2 in range(size):
                    Frequency[i,j]+=Figure[i2,j2]*np.exp(-1j*2*np.pi/m*((i2-size_mean)*(i-m_mean)+(j2-size_mean)*(j-m_mean)
                                                                    ))/N      
    return Frequency

def iFFT(F,m):
    size=F.shape[1]
    size_m=(size-1)/2
    m_mean=(m-1)/2
    Image=np.zeros((m,m))*1j
    for i in range(m):
        for j in range(m):
            for i2 in range(size):
                for j2 in range(size):
                    Image[i,j]+=F[i2,j2]*np.exp(1j*2*np.pi/size*((i2-size_m)*(i-m_mean)+(j2-size_m)*(j-m_mean)
                                                                    ))      
    return Image



def conv(image,kernel):
    image_size=image.shape  #2 dimension
    kernel_size=kernel.shape #3 dimension
    nmax=kernel_size[0]
    ymax=image_size[0]-kernel_size[1]+1
    xmax=image_size[1]-kernel_size[2]+1
    kernel_result=np.zeros((kernel_size[0],ymax,xmax))            
    for n in range(nmax):
        for i in range(ymax):
            for j in range(xmax):
                kernel_result[n,i,j]=np.sum(image[i:i+kernel_size[1],j:j+kernel_size[2]]*kernel[n])
    return kernel_result

def padding(image,ytop,ybottom,xtop,xbottom):
    image_size=image.shape
    ymax=image_size[0]+ytop+ybottom
    xmax=image_size[1]+xtop+xbottom    
    result=np.zeros((ymax,xmax))
    result[ytop:(ymax-ybottom),xtop:(xmax-xbottom)]=image
    return result
    
#max_pooling
def pooling(image,y,x):
    image_size=image.shape
    ymax=int(image_size[0]/y)
    xmax=int(image_size[1]/x)
    if (np.round(image_size[0]/y))!=(image_size[0]/y):
        print('please padding ',(image_size[0]-np.round(image_size[0]/y))*y,' in y direction')
        return 0
    if (np.round(image_size[1]/x))!=(image_size[1]/x):
        print('please padding ',(image_size[1]-np.round(image_size[1]/x))*x,' in x direction')
        return 0
    result=np.zeros((ymax,xmax))
    for i in range(ymax):
        for j in range(xmax):
            buff=image[(i*y):((i+1)*y),(j*x):((j+1)*x)]
            result[i,j]=np.max(buff)
    return result

def nonzero_padding(image,ytop,ybottom,xtop,xbottom):
    image_size=image.shape
    ymax=image_size[0]+ytop+ybottom
    xmax=image_size[1]+xtop+xbottom    
    result=np.zeros((ymax,xmax))+0.0001
    result[ytop:(ymax-ybottom),xtop:(xmax-xbottom)]=image
    return result


def connect(image):
    #image can be multi-demension
    result=image.reshape(image.size)
    return result

def forward(incoming,weight):
    #imcoming is 1D, weight is 2D
    weight_size=weight.shape
    if weight_size[0]!=len(incoming):
        print('matrix dimension mismatch')
        return 0
    result=np.dot(incoming,weight)
    return result

def onehot(numbers):
    size=len(numbers)
    onehot=np.zeros((size,10))
    for i in range(size):
        onehot[i,int(numbers[i])]+=1
    return onehot

def drop(x,y):
    size=x.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if np.random.rand(1)<y:
                x[i,j]=0
    return x

def softmax(numbers):
    expsum=np.sum(np.exp(-numbers))
    return np.exp(-numbers)/expsum

def sigmoid(x,d=0):

    if d==0:
        return 1/(1+np.e**(-x))
    else:
        return x*(1-x)
    
def norm(x):
    size=x.shape
    maxi=np.max(x)
    mini=np.min(x)
    y=np.zeros((size[0],size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            y[i,j]=(x[i,j]-mini)/(maxi-mini)
    return y


# In[1]:


#set kernel 3x3 image

def create_kernel():
    

    kernel=np.zeros((6,3,3))

    
    for i in range(3):
        for j in range(3):
            if i>(2-j):
                kernel[0][i,j]=-1
            if i==(2-j):
                kernel[0][i,j]=0
            if i<(2-j):
                kernel[0][i,j]=1



    for i in range(3):
        for j in range(3):
            if i>j:
                kernel[1][i,j]=-1
            if i==j:
                kernel[1][i,j]=0
            if i<j:
                kernel[1][i,j]=1

    

    for i in range(3):
        for j in range(3):
            if i==0:
                kernel[2][j,i]=-1
            if i==1:
                kernel[2][j,i]=0
            if i==2:
                kernel[2][j,i]=1

    for i in range(3):
        for j in range(3):
            if i==0:
                kernel[3][i,j]=-1
            if i==1:
                kernel[3][i,j]=0
            if i==2:
                kernel[3][i,j]=1

    kernel[4]=np.ones((3,3))
    for i in range(3):
        for j in range(3):
            if ((i==0)or(i==2))&((j==0)or(j==2)):
                kernel[4][i,j]=0
    kernel[4][1,1]=-4


    kernel[5]=np.ones((3,3))

    return kernel


# In[2]:


#load the train set: im_train & lb_train
def train_set():
    
    f=gzip.open('train-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data1 = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data1 = data1.reshape(num_images, image_size, image_size, 1)


    im_train = np.asarray(data1).squeeze()


    f=gzip.open('train-labels-idx1-ubyte.gz','r')

    image_size = 1
    num_images = 60000

    f.read(8)
    buf = f.read(image_size * image_size * num_images)
    data2 = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    lb_train = data2.reshape(num_images)

    return im_train,lb_train


# In[3]:


#load the test set: im_test & lb_test
def test_set():
    f=gzip.open('t10k-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data3 = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data3 = data3.reshape(num_images, image_size, image_size, 1)


    im_test = np.asarray(data3).squeeze()


    f=gzip.open('t10k-labels-idx1-ubyte.gz','r')

    image_size = 1
    num_images = 10000

    f.read(8)
    buf = f.read(image_size * image_size * num_images)
    data4 = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    lb_test = data4.reshape(num_images)

    return im_test,lb_test




