#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:00:42 2020

@author: fangchenfeng
"""

from __future__ import division
import numpy as np
import numpy.matlib
import scipy as sp
from munkres import Munkres

def conv_sparse(X,S,A,max_it,stop_criteria,lambda_max,lambda_final):
    """
    This function does the separation for one frequency
    """
    # updating S
    N = np.size(S,0)
    S_old = S
    cost_it = np.zeros((max_it))
    Nbzer_it = np.zeros((max_it))
    lambda_v = np.logspace(np.log10(lambda_max),np.log10(lambda_final),max_it)
    for it in np.arange(max_it):
        lambda_this = lambda_v[it]
        cost_it[it]=np.linalg.norm(X - np.dot(A,S),ord = 'fro')
        Nbzer_it[it] = 100-100*np.count_nonzero(S)/S.size
        # Ths Lipschitz constant
        L = np.linalg.norm(A,ord=2)**2
        grad = np.dot(-np.conjugate(A.transpose()),(X - np.dot(A,S)))
        S = prox_l1(S - grad/L,lambda_this/L)
    
        # dealing with NaN or Inf
        S[np.isnan(S)]=0
        S[np.isinf(S)]=0
    
        # updating A
        A = np.dot(X,np.conjugate(S.transpose()))
        for ng in np.arange(N):
            A[:,ng]=A[:,ng]/np.linalg.norm(A[:,ng],ord=2)
            
        A[np.isinf(A)]=0
        
        # stoping criteria of convergence
        err = np.linalg.norm(S-S_old,ord='fro')
        if err<stop_criteria:
            print('Convergence reached')
            break
        
        
    
    return S, A, cost_it, Nbzer_it

    
def prox_l1(dccS,lambda_this):
    # This is function perform the soft-thresholding
    arg_dccS = np.zeros(dccS.shape)
    arg_dccS = np.divide(dccS,np.abs(dccS))
    arg_dccS[np.isinf(dccS)]=0
    
    dccS_abs = np.abs(dccS)
    dccS_dif = dccS_abs-lambda_this
    dccS_dif[np.where(dccS_dif<0)]=0
    dccS = np.multiply(arg_dccS,dccS_dif)
    
    return dccS


def PowRatioA(S,A):
    """
    This function calculate the power ratio given a certain t-f plan. The size of S is Mtr*Ntr*N
    The size of A is Mtr*M*N
    """
    Mtr, Ntr, N = S.shape
    PR = np.zeros(S.shape)
    M = np.size(A,1)
    
    for mtrg in np.arange(Mtr):
        # Get the source for one frequency
        S_temp = np.squeeze(S[mtrg,:,:]).transpose()
        PR_temp = np.zeros((N,Ntr))
        
        for ng in np.arange(N):
            for mg in np.arange(M):
                PR_temp[ng,:] = PR_temp[ng,:] + np.abs(A[mtrg,mg,ng]*S_temp[ng,:])**2
                
        PR_all = np.sum(PR_temp,axis=0)
        
        for ng in np.arange(N):
            PR[mtrg,:,ng] = np.divide(PR_temp[ng,:],PR_all)
            
    PR[np.isinf(PR)]=0
    PR[np.isnan(PR)]=0
    
    return PR

def Permutation_first_method(PR,S,A,max_it):
    """
    This function does the correlation-based permutation
    """
    
    
def Permutation_correlation_oracle(PR,dccS,dccA,dccSo):
    """
    This function does the oracle permutation based on the correlation interfreq-
    uency.
    Input:  PR: Mtr*Ntr*N: the Power Ratio
            dccS: Mtr*Ntr*N: analysis coefficients of the estimated sources
            dccA: Mtr*M*N: the mixing coefficient
            dccSo: Mtr*Ntr*N: oracle coefficients of the sources
    """
    Mtr = np.size(PR,0)
    N = np.size(dccS,2)
    M = np.size(dccA,1)
    # the initial order is 1 2 3..for every frequency mtrg
    perm = np.matlib.repmat(np.arange(N),Mtr,1)
    # I calculate now the original ratio power
    PRo = PowRatioA(dccSo,np.ones((Mtr,M,N)))
     
    dccS_reorder = dccS
    dccA_reorder = dccA
    
    print('Begin oracle permutation based on original sources')
    
    for mtrg in np.arange(Mtr):
        # I calculate the correlation matrix between the estimated PR and the original PR
        PRo_temp = np.squeeze(PRo[mtrg,:,:]).transpose()
        PR_temp = np.squeeze(PR[mtrg,:,:]).transpose()
        PR_temp, order_temp = Put_in_order(PRo_temp,PR_temp)
        PR[mtrg,:,:]=PR_temp.transpose()
        # I do the same permutation for dccS
        dccS_temp = np.squeeze(dccS[mtrg,:,:])
        dccS_reorder[mtrg,:,:] = dccS_temp[:,order_temp]
        # I do the same permutation for dccA
        dccA_temp = np.squeeze(dccA[mtrg,:,:])
        dccA_reorder[mtrg,:,:] = dccA_temp[:,order_temp]
        # I do the same thing for perm
        perm_temp = perm[mtrg,:]
        perm[mtrg,:] = perm_temp[order_temp]
        
    return PR, dccS_reorder, dccA_reorder, perm    
    
        
        
def Put_in_order(C,PR):
    """
    This function tries to range the order of PR so that it will maximise the correlation
    with C (centroid)
    """
    # I first do the normalisation and supprime the mean value
    N, Ntr = C.shape
    PRo = PR
    for ng in np.arange(N):
        C[ng,:] = C[ng,:]-np.mean(C[ng,:])
        if np.std(C[ng,:])!=0:
            C[ng,:] = C[ng,:]/np.std(C[ng,:])
        else:
            C[ng,:] = np.zeros(C[ng,:].shape())
            
        PR[ng,:] = PR[ng,:]-np.mean(PR[ng,:])
        if np.std(PR[ng,:])!=0:
            PR[ng,:]=PR[ng,:]/np.std(PR[ng,:])
        else:
            PR[ng,:] = np.zeros((1,Ntr))
            
    # Now we construct the correlation matrix
    P = np.dot(C,PR.transpose())
    P = np.ones(P.shape) - P
    # Use the munkres algorithm
    m = Munkres()
    indexes = m.compute(P)
    order = np.asarray(indexes)[:,1]
    
    PR = PRo[order,:]
    
    return PR, order
                
    
    
    