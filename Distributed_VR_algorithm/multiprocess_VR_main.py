from __future__ import division

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from VR_algorithm import Multiprocess_VR_agent
from cost_func import *
from read_data import *

from multiprocessing import Process, Pipe
import networkx as nx

def generate_topology(N_node=20, prob=0.25):
    G=nx.random_geometric_graph(N_node, prob)
    pos=nx.get_node_attributes(G,'pos')

    dmin=1
    ncenter=0
    for n in pos:
        x,y = pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
        
    p=nx.single_source_shortest_path_length(G,ncenter)

    return G

def sk_train(X, y, C=1):
    lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-7, max_iter=300)
    lr.fit(X, np.squeeze(y))
    return lr.coef_.T

def generate_agent(G, X, y, N_data_boundary, w_star, cost_model, **kwargs):
    edges = set() # use set to avoid bi-direction edges create two pipes
    for i, edge in enumerate(G.adjacency_list()):
        for e in edge:
            edges.add((i,e)) if i<e else edges.add((e,i))
        
    # each edge will associate with one pipe
    pipe_dict = {}
    for e in edges:
        pipe_dict[e] = Pipe()

    agent_list = []
    for n, edge in enumerate(G.adjacency_list()):
        pipe_list = []
        for e in edge:
            if (n, e) in pipe_dict:
                pipe_list.append(pipe_dict[(n,e)][0])
            elif (e, n) in pipe_dict:
                pipe_list.append(pipe_dict[(e,n)][1])

        data_range = np.arange(N_data_boundary[n], N_data_boundary[n+1])
        agent_list.append( \
            Multiprocess_VR_agent(X[data_range], y[data_range], w_star, cost_model, pipe_list, \
                        rho = kwargs.get('rho', 1e-3), name = n)
            )
    return agent_list


if __name__ == '__main__':
    N_agent = 10
    G = generate_topology(N_agent, prob=.6)

    # X, y = read_cifar()

    # X, y = gen_lr_data(1000, 28*28)
    # norm = sklearn.preprocessing.Normalizer()
    # X = norm.fit_transform(X)
    X, y = read_mnist()

    print (X.shape, y.shape)
    X, y, N_data_boundary, N_data_agent_mean = split_data(X, y, N_agent, even=True)

    C, N_epoch = 1, 1
    rho = 1/X.shape[0]/C
    w_star = sk_train(X,y,C)

    agent_list = generate_agent(G, X, y, N_data_boundary, w_star, logistic_regression, rho=rho)
    max_ite = 10000 #N_epoch*N_data_agent_mean

    mu = 8.0 #0.7
    Data_dist = np.array([a.N for a in agent_list])
    mu_each = mu*Data_dist/sum(Data_dist)*N_agent
    print (mu_each)
    ps = [Process(target=a.train, args=(mu_each[i], max_ite, 'SVRG', 'Diffusion',), \
                    kwargs = {'err_per_iter': 10, 'metric': 'MSD', 'using_sgd': 2}) \
            for i, a in enumerate(agent_list)]

    [p.start() for p in ps]

    [p.join() for p in ps]

    print ('Done')