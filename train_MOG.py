import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *


def train_MOG(X_phoneme, k):
    X = X_phoneme.copy()
    N = X.shape[0]
    D = X.shape[1]
    p = np.ones((k))/k
    random_indices = np.floor(N*np.random.rand((k)))
    random_indices = random_indices.astype(int)
    mu = X[random_indices,:] 
    s = np.zeros((k,D,D)) 
    n_iter = 100
    for i in range(k):
        cov_matrix = np.cov(X.transpose())
        s[i,:,:] = cov_matrix/k
    Z = np.zeros((N,k)) # shape Nxk
    for t in range(n_iter):
        print('Iteration {:03}/{:03}'.format(t+1, n_iter))
        Z = get_predictions(mu, s, p, X)
        print(Z)
        Z = normalize(Z, axis=1, norm='l1')
        for i in range(k):
            mu[i,:] = np.matmul(X.transpose(),Z[:,i]) / np.sum(Z[:,i])
            mu_i = mu[i,:]
            mu_i = np.expand_dims(mu_i, axis=1)
            mu_i_repeated = np.repeat(mu_i, N, axis=1)
            X_minus_mu = (X.transpose() - mu_i_repeated)**2
            res_1 = np.squeeze( np.matmul(X_minus_mu, np.expand_dims(Z[:,i], axis=1)))/np.sum(Z[:,i])
            s[i,:,:] = np.diag(res_1)
            p[i] = np.mean(Z[:,i])

    return p, mu, s