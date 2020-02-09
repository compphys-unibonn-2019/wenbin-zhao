#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import linalg


# In[8]:


# 1+1 toy model


# In[9]:


# generate SU(2) matrix
x = np.zeros(3)
x = np.random.uniform(-1,1,3)
z1 = complex(x[1],x[2])
z2 = complex(x[1],-x[2])
H = np.matrix([[x[0],z1],[z2,-x[0]]])
M = linalg.expm(0.24j*H)


# In[10]:


t = 8
x = 8
Link = np.zeros((t,x,2,2,2),dtype = complex)  # the third index is 0 or 1, corresponding to time and space link


# In[11]:


for t in range(8):    # innitial Link 
    for y in range(8): # use y to label position,  x is used already
        for i in range(2):
            x = np.zeros(3)
            x = np.random.uniform(-1,1,3)
            z1 = complex(x[1],x[2])
            z2 = complex(x[1],-x[2])
            H = np.matrix([[x[0],z1],[z2,-x[0]]])
            Link[t,y,i,:,:] = linalg.expm(1j*H) # Link is an array not a matrix now!


# In[12]:


def newSU2(): # a SU2 matrix around I
    x = np.zeros(3)
    x = np.random.uniform(-1,1,3)
    z1 = complex(x[1],x[2])
    z2 = complex(x[1],-x[2])
    H = np.matrix([[x[0],z1],[z2,-x[0]]])
    M = linalg.expm(0.24j*H)
    return M


# In[13]:


def P10(Link,t,x):  # the smallest wilson loop
    U0tx = np.matrix(Link[t,x,0,:,:]) #   (t,x)--(t+a,x)  trans2D array to Matrix  so getH defined.
    U1tx = np.matrix(Link[(t+1)%8,x,1,:,:])    #(t+a,x)--(t+a,x+a)
    U0txh = (np.matrix(Link[(t+1)%8,(x+1)%8,0,:,:])).getH() #(t+a,x+a) --- (t,x+a)
    U1txh = (np.matrix(Link[t,(x+1)%8,1,:,:])).getH() #  (t,x+a)---(t,x)
    P = U0tx@U1tx@U0txh@U1txh  # @ is matrix multiplication for 2d
    P = np.trace(P)
    return 1/3*P.real


# In[14]:


def SWil(Link,beta): # wilison action with coupling beta
    S = 0
    for t in range(8):
        for x in range(8):
            S = S - beta * P10(Link,t,x)
    return S        


# In[15]:


def update(Link,t,x,beta):  # update one point
    for i in range(10):  #update each point several(10) times before move to another point
        if i == 0 :
            old = -beta*( P10(Link,t,x)+P10(Link,(t+1)%8,x)+P10(Link,t,(x+1)%8)+ P10(Link,t-1,x)+P10(Link,t,x-1) )
            oldLink = Link[t,x,:,:,:]    # waste some memory space 
            Link[t,x,1,:,:] = newSU2()@Link[t,x,1,:,:]
            Link[t,x,0,:,:] = newSU2()@Link[t,x,0,:,:] # change link on time and position at the same time ?? should do separately?
            new = -beta*( P10(Link,t,x)+P10(Link,(t+1)%8,x)+P10(Link,t,(x+1)%8)+ P10(Link,t-1,x)+P10(Link,t,x-1) )
            change = new - old 
            if change>0 and np.exp(-change) < np.random.random():
                Link[t,x,:,:,:] = oldLink
            else:    # if Link is changed
                old = new
        else:
            oldLink = Link[t,x,:,:,:]
            Link[t,x,1,:,:] = newSU2()@Link[t,x,1,:,:]
            Link[t,x,0,:,:] = newSU2()@Link[t,x,0,:,:] 
            new = -beta*( P10(Link,t,x)+P10(Link,(t+1)%8,x)+P10(Link,t,(x+1)%8)+ P10(Link,t-1,x)+P10(Link,t,x-1) )
            change = new - old 
            if change>0 and np.exp(-change) < np.random.random():
                Link[t,x,:,:,:] = oldLink
            else:
                old = new
    return Link


# In[16]:


def W(Link,t,r):    # r is position(1), t is time(0)  
    W = np.zeros(0)
    Ut = np.zeros((t,2,2),dtype = complex)
    Uth = np.zeros((t,2,2),dtype = complex)
    Ux = np.zeros((r,2,2),dtype = complex)
    Uxh = np.zeros((r,2,2),dtype = complex)
    U0tx = np.matrix([[1,0],[0,1]],dtype = complex) # results of (0,0)----(t,0)
    U1tx = np.matrix([[1,0],[0,1]],dtype = complex) # results of (t,0)----(t,r)
    U0txh = np.matrix([[1,0],[0,1]],dtype = complex)# results of (t,r)----(0,r)
    U1txh = np.matrix([[1,0],[0,1]],dtype = complex) # results of (0,r)----(0,0)
    for i in range(t):
        Ut[i,:,:] = Link[i,0,0,:,:]
        Uth[t-i-1,:,:] = (np.matrix(Link[i,r-1,0,:,:])).getH()
    for j in range(r):
        Ux[j,:,:] = Link[t-1,j,1,:,:]
        Uxh[r-j-1,:,:] = (np.matrix(Link[0,j,1:,:])).getH()
    for i in range(t):
        U0tx = U0tx @ np.matrix(Ut[i,:,:])
        U0txh = U0txh @ np.matrix(Uth[i,:,:])
    for j in range(r):
        U1tx = U1tx @ np.matrix(Ux[j,:,:])
        U1txh = U1txh@ np.matrix(Uxh[j,:,:])
        
    P = U0tx@U1tx@U0txh@U1txh  # @ is matrix multiplication for 2d
    P = np.trace(P)
    return 1/3*P.real


# In[17]:


for N in range(100):
    for t in range(8):
        for x in range(8):
            update(Link,t,x,5.5)
    print ('\r', N/100,end='')  # tracking the process


# In[23]:


SWil(Link,5.5)


# In[24]:


N = 50
Ncor = 50
t0 = 8
x0 = 8 
W61 = np.zeros(N)
W71 = np.zeros(N)
W62 = np.zeros(N)
W72 = np.zeros(N)

for i in range(N):
    for j in range(Ncor):
        for t in range(t0):
            for x in range(x0):
                update(Link,t,x,5.5)
    W61[i] = W(Link,6,1)
    W71[i] = W(Link,7,1)
    W62[i] = W(Link,6,2)
    W72[i] = W(Link,7,2)
    print ('\r', i/N,end='')


# In[25]:


V1 = W61/W71


# In[26]:


V2 = W62/W72


# In[27]:


V1.mean()


# In[28]:


V2.mean()


# In[ ]:




