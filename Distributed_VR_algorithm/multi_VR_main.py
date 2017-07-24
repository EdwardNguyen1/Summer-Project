from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from VR_algorithm import Dist_VR_agent
from cost_func import *
from read_data import *

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

def generate_network(N_node=20, method='metropolis_rule', **kwargs):
    '''
        Wrap function to generate topology. 
        Then by using avg or metropolis rule to generate the combination matrix
        current support key words:   prob----------connected probability

        method only support 'average_rule' and 'metropolis_rule' now
    '''
    
    # start with 1 because assuming every node has self-loop
    indegree = np.ones((N_node,1))
    G = generate_topology(N_node, kwargs.get('prob', 0.25))
    for edge in G.edges():
      indegree[edge[0]] += 1
      indegree[edge[1]] += 1

    def avg_rule(G, indegree):
        N_node = indegree.shape[0]
        A = np.zeros((N_node,N_node))
        for e1, e2 in G.edges():
            A[e1,e2] = 1./indegree[e2]
            A[e2,e1] = 1./indegree[e1]

        for i in range(N_node):
            A[i,i] = 1. - np.sum(A[:,i])
        return A

    def metropolis_rule(G, indegree):
        N_node = indegree.shape[0]
        A = np.zeros((N_node,N_node))
        for e1, e2 in G.edges():
            A[e1,e2] = 1./max(indegree[e1], indegree[e2])
            A[e2,e1] = 1./max(indegree[e1], indegree[e2])

        for i in range(N_node):
            A[i,i] = 1. - np.sum(A[:,i])
        return A 

    option = {'average_rule': avg_rule,
              'metropolis_rule': metropolis_rule}

    if method not in option:
        print ('Currently, only support "average_rule" and "metropolis_rule"')

    return option[method](G, indegree) 


def sk_train(X, y, C=1):
    lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-7, max_iter=300)
    lr.fit(X, np.squeeze(y))
    return lr.coef_.T

def generate_agent(X, y, N_data_boundary, w_star, cost_model, **kwargs):
    agent_list = []
    for agent in range(len(N_data_boundary)-1):
        data_range = np.arange(N_data_boundary[agent],N_data_boundary[agent+1])
        agent_list.append( \
            Dist_VR_agent(X[data_range], y[data_range], w_star, cost_model, rho = kwargs.get('rho', 1e-3))
            )
    return agent_list

def train(agent_list,  A_bar, mu, max_ite, method = 'AVRG', dist_style = 'Diffusion', **kwargs):
    err = []
    # step-size need to adjust
    N_agent = len(agent_list)
    Data_dist = np.array([a.N for a in agent_list])
    mu_each = mu*Data_dist/sum(Data_dist)*N_agent
    print ('mu_each: ', mu_each)

    err_per_iter = kwargs.get('err_per_iter', int(np.mean(Data_dist)))
    for ite in range(max_ite):
        if (ite % 1000) == 0:
            print ("Calculating iteration: ", ite)

        # adapt and correct
        for n, agent in enumerate(agent_list):
            agent.adapt(mu_each[n], ite, method, dist_style, **kwargs)
            agent.correct(ite, dist_style)

        # combination
        for n, agent in enumerate(agent_list):
            weight_list = [A_bar[n, i] for i in range(N_agent) if A_bar[n, i]!=0]
            if dist_style == 'Diffusion':
                w_list = [agent_list[i].phi.copy() for i in range(N_agent) if A_bar[n, i]!=0]
            elif dist_style == 'EXTRA':
                w_list = [agent_list[i].cost_model.w.copy() for i in range(N_agent) if A_bar[n, i]!=0]
            else:
                print ('Not support %s style' % dist_style)
            agent.combine(weight_list, w_list, style=dist_style)


        if ite % err_per_iter == 0:
            err_each = 0
            for agent in agent_list:
                err_each += agent.Performance(metric=kwargs.get('metric','MSD'))
            err.append(err_each/N_agent)

    return err
            


if __name__ == '__main__':
    N_agent = 10

    # X, y = read_cifar()

    # X, y = gen_lr_data(1000, 28*28)
    # norm = sklearn.preprocessing.Normalizer()
    # X = norm.fit_transform(X)
    X, y = read_mnist()

    X, y, N_data_boundary, N_data_agent_mean = split_data(X, y, N_agent)

    C, N_epoch = 1, 15
    rho = 1/X.shape[0]/C
    w_star = sk_train(X,y,C)

    # print(N_data_boundary)

    agent_list = generate_agent(X, y, N_data_boundary, w_star, logistic_regression, rho=rho)

    A = generate_network(N_agent, method='metropolis_rule', prob=0.6)
    A_bar = (A + np.eye(N_agent))/2
    print (A_bar)

    max_ite = 10000 #N_epoch*N_data_agent_mean
    mu = 9
    err = train(agent_list, A_bar, mu, max_ite, method = 'AVRG', dist_style = 'Diffusion', replace=True, using_sgd=2)

    plt.semilogy(err)
    plt.show()


