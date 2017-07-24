from __future__ import division

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from VR_algorithm import ZMQ_VR_agent
from cost_func import *
from read_data import *
import zmq


def sk_train(X, Y, C=1):
    lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-7, max_iter=300, multi_class='multinomial', solver='sag')
    y = np.argmax(Y, axis = 1) # change one-hot back to target label
    lr.fit(X, y)
    return lr.coef_.reshape(-1,1)

def read_mnist_wrap(mask_label = list(range(10)), test_pec = 0.5):
    X, Y = read_mnist(datatype = 'multiclass', mask_label = mask_label)
    N = X.shape[0]

    N_train_prctg, N_val_prctg = 0.8*(1-test_pec), 0.2*(1-test_pec)
    N_train = int(N * N_train_prctg)
    N_val = int(N * N_val_prctg)

    X_train, Y_train = X[0:N_train, :], Y[0:N_train, :]
    X_val, Y_val = X[N_train: N_train+N_val, :], Y[N_train: N_train+N_val, :]
    X_test, Y_test = X[N_train+N_val:N+1, :], Y[N_train+N_val:N+1, :]

    y_train, y_val, y_test = np.argmax(Y_train, axis = 1), np.argmax(Y_val, axis = 1), np.argmax(Y_test, axis = 1)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test 

if __name__ == '__main__':
    N_agent = 2

    X_train, Y_train, X_val, Y_val, X_test, Y_test = read_mnist_wrap(list(range(5,10)), test_pec=0.5)
    X_train2, Y_train2, _, _, _, _ = read_mnist_wrap(list(range(0,5)), test_pec=0.5)

    N = X_train.shape[0] + X_train2.shape[0]
    D, Class = X_train.shape[1], Y_train.shape[1]
    C = 1
    rho = 1/N/C

    print ("calculating w_star now...")
    w_star = sk_train(np.vstack([X_train,X_train2]), np.vstack([Y_train,Y_train2]), C)

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    port = '6666'
    socket.connect("tcp://localhost:%s" % port)
    print ("Start connect.....")

    # server get first half data
    agent = ZMQ_VR_agent(X_train, Y_train, w_star, soft_max, [socket], rho = rho, name = 0)
    mu = 2.0
    max_ite = 50000
    kwargs = {'err_per_iter': 20, 'metric': 'MSD', 'using_sgd': 2}
    agent.train(mu, max_ite, 'AVRG', 'Diffusion', **kwargs)

    socket.close()
