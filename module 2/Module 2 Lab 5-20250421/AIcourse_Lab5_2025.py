# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:42:14 2023

@author: Eric (Mint Lab)
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def state(x):
    # function to convert throughput to state
    ''' please write your state '''
    s = math.ceil(x * state_level/Rmax)
    return s

def reward(P, Nap, h):
    #Calculates the total throughput given the transmit powers, number of APs, and channel matrix.
    out = 0
    for i in range(Nap):
        for j in range(Nap):
            S = P[i]*h[i,i]
            I = 0
            N = 1
            if i != j:
                I += P[j]*h[j,i]
        ''' please write your reward '''
        out += math.log2(1 + (S/(I + N)))
    return out

#%%
# initialize variables
Nap = 5 # assume same number of APs and UEs
Gain = 10
h = np.random.rand(Nap,Nap)*Gain # channel dimension: AP x UE
pts = 10 # for quantization
P_max = 10
P = np.linspace(1, P_max, pts)
at_level = len(P)**Nap
state_level = 10 # for quantization
eta = 0.05 # learning rate
gamma = 0.8 # next important
iter = int(1e6)

#%%
R = 0
for i in range(int(1e6)):
    atmp = np.ceil(np.random.rand(1,Nap)*pts)
    Power = np.zeros(Nap)
    for k in range(Nap):
        Power[k] = P[int(atmp[0,k]-1)]
    Ttmp = reward(Power, Nap, h)
    if Ttmp > R:
        R = Ttmp
Rmax = R
#%%

Q = [np.zeros((pts,state_level)) for _ in range(Nap)] # each AP has a Q-table
s = np.ones(iter+1, dtype=int)
act = np.ones((Nap,iter), dtype=int)
throughput = np.ones(iter)

for i in range(iter):
    if random.random() < 0.5 or i == 0: # random action
        temp = np.ceil(np.random.rand(1,Nap)*pts)
        a = temp[0,:].astype(int) # 自己改的
        Power = np.zeros(Nap)
        for k in range(Nap):
            Power[k] = P[int(a[k]-1)]
    else: # optimum action
        a = np.zeros(Nap, dtype=int)
        for k in range(Nap):
            '''How to chose the action?'''
            a[k] = np.argmax(Q[k][:,s[i]-1])+1
        Power = np.zeros(Nap)
        for k in range(Nap):
            Power[k] = P[int(a[k]-1)]
    act[:,i] = a
    '''Reward'''
    throughput[i] = reward(Power, Nap, h)
    '''State'''
    s[i+1] = state(throughput[i])
    if s[i+1] <= 0:
        s[i+1] = 1
    elif s[i+1] > state_level-1:
        s[i+1] = state_level-1
    penalty = np.zeros(Nap)
    for k in range(Nap):
        if Power[k] >= P_max:
            penalty[k] = 1
    for k in range(Nap):
        '''Please write how your Q-table updates'''
        Q[k][a[k]-1, s[i]-1] = Q[k][a[k]-1, s[i]-1] + eta * (throughput[i] + gamma * np.max(Q[k][:, s[i+1]-1]) - Q[k][a[k]-1, s[i]-1]) - penalty[k]

#%%
plt.figure() 
ax1 = plt.subplot(1,2,1)
ax1.plot(s)
ax1.set_xlabel('State')
ax2 = plt.subplot(1,2,2)
ax2.plot(throughput)
ax2.set_xlabel('throughput')
plt.show()

#%%
# Batch processing
batch = iter//200;
SS = np.ones(iter//batch)
TT = np.ones(iter//batch)
AA = np.ones((Nap, iter//batch))

for i in range(int(iter/batch)):
    SS[i] = np.mean(s[(i-1)*batch+1:i*batch])
    for j in range(Nap):
        AA[j,i] = np.mean(act[j,(i-1)*batch+1:i*batch])
    TT[i] = np.mean(throughput[(i-1)*batch+1:i*batch]);

plt.figure()    
ax1 = plt.subplot(1,3,1)
ax1.plot(SS)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Averaged State')
ax1.legend(['State'])
ax2 = plt.subplot(1,3,2)
for i in range(Nap):
    ax2.plot(AA[i,:])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Averaged Action')
ax2.legend(["AP1","AP2","AP3","AP4","AP5"])
ax3 = plt.subplot(1,3,3)
ax3.plot(TT)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Averaged Sum Rate')
ax3.legend(['Sum Rate (bps/Hz)'])
plt.show()