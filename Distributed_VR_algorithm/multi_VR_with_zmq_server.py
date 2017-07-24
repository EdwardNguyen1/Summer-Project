from __future__ import division

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from VR_algorithm import ZMQ_VR_agent
from cost_func import *
from read_data import *
import zmq


def sk_train(X, y, C=1):
    lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-7, max_iter=300)
    lr.fit(X, np.squeeze(y))
    return lr.coef_.T

if __name__ == '__main__':
    N_agent = 2

    # X, y = read_cifar()

    # X, y = gen_lr_data(1000, 28*28)
    # norm = sklearn.preprocessing.Normalizer()
    # X = norm.fit_transform(X)
    X, y = read_mnist()

    # make data evenly distributed
    X = X[:-1] if X.shape[0]%2 else X
    y = y[:-1] if y.shape[0]%2 else y
    N = X.shape[0]
    print (X.shape, y.shape)

    C, N_epoch = 1, 1
    rho = 1/X.shape[0]/C
    w_star = sk_train(X,y,C)

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    port = '6666'
    socket.bind("tcp://*:%s" % port)
    print ("Start listening.....")

    # server get first half data
    agent = ZMQ_VR_agent(X[:N//2], y[:N//2], w_star, logistic_regression, [socket], rho = rho, name = 0)
    mu = 2.0
    max_ite = 40000
    kwargs = {'err_per_iter': 20, 'metric': 'MSD', 'using_sgd': 2}
    agent.train(mu, max_ite, 'SVRG', 'Diffusion', **kwargs)

    socket.close()
