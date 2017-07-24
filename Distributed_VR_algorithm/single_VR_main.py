import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from VR_algorithm import VR_algorithm
from cost_func import logistic_regression, least_square
from read_data import *

def sk_train(X, y, C=1):
	lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-7, max_iter=300)
	lr.fit(X, np.squeeze(y))
	return lr.coef_.T

if __name__ == '__main__':
	X,y = read_cifar()
	# print (X.shape, y.shape)

	C, N_epoch = 1, 15
	rho = 1/X.shape[0]/C
	w_star = sk_train(X,y,C)
	# print (w_star)
	params= {'using_sgd': 1,
         	 'epoch_per_FG': 2,
         	 'minibatch': 1}

	svrg = VR_algorithm(X, y , w_star, logistic_regression, rho = rho)
	MSD, ER = svrg.train(N_epoch=N_epoch, mu=1.0, method='SVRG', **params, replace = True)

	avrg = VR_algorithm(X, y , w_star, logistic_regression, rho = rho)
	MSD, ER = avrg.train(N_epoch=N_epoch, mu=.5, method='AVRG', **params, replace = False)

	saga = VR_algorithm(X, y , w_star, logistic_regression, rho = rho)
	MSD, ER = saga.train(N_epoch=N_epoch, mu=0.6, method='SAGA', **params, replace = True)

	plt.semilogy(np.arange(N_epoch)*2+2,avrg.ER,'b^-',lw=1.5)
	plt.semilogy(np.arange(N_epoch)*2.5+2.5,svrg.ER,'ks-',lw=1.5)
	plt.semilogy(np.arange(N_epoch)+1,saga.ER,'ro-',lw=1.5)

	plt.grid('on')
	plt.legend(['AVRG','SVRG','SAGA','SAGA+RR'],loc=3)
	plt.xlabel('Gradients/N')
	plt.ylabel('$\mathrm{\mathbb{E} } J(w^k_i)-J(w^\star)$')
	plt.show()


	
