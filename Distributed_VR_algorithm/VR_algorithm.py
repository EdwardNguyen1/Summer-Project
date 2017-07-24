import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time


class VR_algorithm():
    def __init__(self, X, y, w_star, cost_model, **kwargs):
        '''
            cost_model should provide partial_gradient and full_gradient etc.
        '''
        self.cost_model = cost_model(X, y, **kwargs) 
        self.w_star = w_star
        self.norm_w_star =  np.sum(np.square(self.w_star))
        self.N, self.M = X.shape[0], X.shape[1]
        self.VR_option = {'SVRG': self.SVRG_step,
                       'AVRG': self.AVRG_step,
                       'SAGA': self.SAGA_step,
                       'SGD': self.SGD_step,
                       'GD': self.GD_step}

    def SVRG_step(self, ite, **kwargs):
        minibatch = kwargs.get('minibatch',1)
        N_by_batch = int(np.floor(self.cost_model.N / minibatch))
        epoch_per_FG = kwargs.get("epoch_per_FG", 2) # particular for SVRG
        using_sgd = kwargs.get('using_sgd',1)

        if ite % N_by_batch == 0:
            self.reorder =  np.random.choice(N_by_batch, N_by_batch, replace=kwargs.get('replace', True))

        if ite % (epoch_per_FG*N_by_batch) == 0:
            self.w_at_start = copy.deepcopy(self.cost_model.w)
            self.grad_full_at_start = self.cost_model.full_gradient()

        idx = self.reorder[ite % N_by_batch]
        selected_idx = np.arange(idx*minibatch, (idx+1)*minibatch)

        grad = self.cost_model.partial_gradient(index=selected_idx)
        grad_at_start = self.cost_model.partial_gradient(index=selected_idx, w_ = self.w_at_start)

        grad_modified = grad - grad_at_start + self.grad_full_at_start if ite>using_sgd*N_by_batch else grad

        return grad_modified

    def AVRG_step(self, ite, **kwargs):
        minibatch = kwargs.get('minibatch',1)
        N_by_batch = int(np.floor(self.cost_model.N / minibatch))
        using_sgd = kwargs.get('using_sgd',1)

        if ite % N_by_batch == 0:
            self.reorder =  np.random.choice(N_by_batch, N_by_batch, replace=kwargs.get('replace', False))

            self.w_at_start = copy.deepcopy(self.cost_model.w)
            self.grad_full_at_start = \
                copy.deepcopy(self.grad_full_at_start_next) if ite != 0 else np.zeros_like(self.cost_model.w)
            self.grad_full_at_start_next = np.zeros(self.grad_full_at_start.shape)


        idx = self.reorder[ite % N_by_batch]
        selected_idx = np.arange(idx*minibatch, (idx+1)*minibatch)

        grad = self.cost_model.partial_gradient(index=selected_idx)
        grad_at_start = self.cost_model.partial_gradient(index=selected_idx, w_ = self.w_at_start)

        grad_modified = grad - grad_at_start + self.grad_full_at_start if ite>using_sgd*N_by_batch else grad

        self.grad_full_at_start_next += (grad / N_by_batch)

        return grad_modified

    def SAGA_step(self, ite, **kwargs):
        minibatch = kwargs.get('minibatch',1)
        N_by_batch = int(np.floor(self.cost_model.N / minibatch))
        using_sgd = kwargs.get('using_sgd',1)


        # SAGA particular: initialize the memory:
        if ite == 0:
            self.grad_at_last =  np.zeros( (self.cost_model.M, self.cost_model.N))
            self.grad_avg = np.zeros( (self.cost_model.M, 1))

        if ite % N_by_batch == 0:
            self.reorder =  np.random.choice(N_by_batch, N_by_batch, replace=kwargs.get('replace', True))

        idx = self.reorder[ite % N_by_batch]
        selected_idx = np.arange(idx*minibatch, (idx+1)*minibatch)

        grad = self.cost_model.partial_gradient(index=selected_idx)

        grad_modified = grad - self.grad_at_last[:,[idx]] + self.grad_avg if ite>using_sgd*N_by_batch else grad

        self.grad_avg = self.grad_avg - (self.grad_at_last[:,[idx]] - grad)/N_by_batch
        self.grad_at_last[:,[idx]] = grad.copy()

        return grad_modified

    def SGD_step(self, ite, **kwargs):
        minibatch = kwargs.get('minibatch',1)
        N_by_batch = int(np.floor(self.cost_model.N / minibatch))

        if ite % N_by_batch == 0:
            self.reorder =  np.random.choice(N_by_batch, N_by_batch, replace=kwargs.get('replace', True))

        idx = self.reorder[ite % N_by_batch]
        selected_idx = np.arange(idx*minibatch, (idx+1)*minibatch)
        grad = self.cost_model.partial_gradient(index=selected_idx)

        return grad

    def GD_step(self, ite, **kwargs):

        return self.cost_model.full_gradient()


    def train(self, N_epoch=10, mu=0.1, method='SVRG', **kwargs):
        self.MSD = []
        self.ER = []

        for ite in range(N_epoch*self.cost_model.N):
            grad_modifed = self.VR_option[method](ite, **kwargs)
            self.cost_model._update_w(grad_modifed, mu)

            if (ite+1) % self.cost_model.N == 0:
                print ('epoch: %d' %((ite+1)/self.cost_model.N))
                err_ = np.sum( (self.cost_model.w - self.w_star)*(self.cost_model.w - self.w_star) ) / self.norm_w_star
                self.MSD.append(err_)
                self.ER.append(self.cost_model.func_value() - self.cost_model.func_value(w_ = self.w_star))

        return self.MSD, self.ER

    
    def soft_threshold(self, delta):
        self.cost_model.w = np.sign(self.cost_model.w)*( np.maximum(np.abs(self.cost_model.w)-delta, 0) )

    def pg_train(self, N_epoch=10, mu=0.1, l1_rho=1e-5, method='SVRG', **kwargs):
        self.MSD = []
        self.ER = []

        for ite in range(N_epoch*self.cost_model.N):
            grad_modifed = self.VR_option[method](ite, **kwargs)
            self.cost_model._update_w(grad_modifed, mu)
            self.soft_threshold(mu * l1_rho)

            if (ite+1) % self.cost_model.N == 0:
                err_ = np.sum( (self.cost_model.w - self.w_star)*(self.cost_model.w - self.w_star) ) / self.norm_w_star
                self.MSD.append(err_)
                self.ER.append(self.cost_model.func_value()+l1_rho*np.sum(np.abs(self.cost_model.w)) \
                                - self.cost_model.func_value(w_ = self.w_star)- l1_rho*np.sum(np.abs(self.w_star)) )

        return self.MSD, self.ER

    def reset(self):
        self.cost_model._reset_w()
        self.grad_at_last =  np.zeros( (self.cost_model.M, self.cost_model.N))
        self.grad_avg = np.zeros( (self.cost_model.M, 1))
        self.grad_full_at_start_next = np.zeros(self.cost_model.w.shape)


class Dist_VR_agent(VR_algorithm):
    '''
        naive implementation for mimic distributed vr algorithm in single-computer(single process)
    '''
    def __init__(self, X, y, w_star, cost_model, **kwargs):
        VR_algorithm.__init__(self, X, y, w_star, cost_model, **kwargs)

        # for distributed algorithm
        self.phi = np.zeros((self.M,1))
        self.psi = np.zeros((self.M,1))
        self.psi_last = np.zeros((self.M,1)) # used for exact diffusion

    def adapt(self, mu, ite, method='AVRG', style='Diffusion', **kwargs):
        self.psi_last = self.psi.copy()
        if style == 'Diffusion':
            self.psi = self.cost_model.w - mu * self.VR_option[method](ite, **kwargs)  
        elif style == 'EXTRA':
            self.psi = self.phi - mu * self.VR_option[method](ite, **kwargs)
        else:
            print ('Not support %s style' % style)
        
    def correct(self, ite, style='Diffusion'):
        if style =='Diffusion':
            self.phi = self.psi - self.psi_last + self.cost_model.w if ite!=0 else self.psi.copy()
        elif style == 'EXTRA':
            self.cost_model.w = self.psi - self.psi_last + self.phi if ite!=0 else self.psi.copy()
        else:
            print ('Not support %s style' % style)
            raise

    def combine(self, weight_list, w_list, style='Diffusion'):
        if len(weight_list) != len(w_list):
            print ("need match!")
            raise

        w_tmp = np.zeros( (self.M,1) )
        for i, weight in enumerate(weight_list):
            w_tmp += weight*w_list[i]
        
        if style == 'Diffusion':
            self.cost_model.w = w_tmp.copy()
        elif style == 'EXTRA':
            self.phi = w_tmp.copy()
        else:
            print ('Not support %s style' % style)


    def Performance(self, metric='MSD', w_=None):
        w = self.cost_model.w if w_ is None else w_

        if metric == 'MSD':
            return np.sum( (w - self.w_star)*(w - self.w_star) ) / self.norm_w_star
        elif metric == 'ER':
            return self.cost_model.func_value(w_ = w) - self.cost_model.func_value(w_ = self.w_star)
        else:
            print ('Unknown metric')
            return None


class multi_VR_agent_self(VR_algorithm):
    '''
        basic self operation for multi-agent VR algorithm, i.e. adapt and correct step and train. 
        The combination step will be pass which will be overwritten by sub-class based on different implement.
    '''
    def __init__(self, X, y, w_star, cost_model, **kwargs):
        VR_algorithm.__init__(self, X, y, w_star, cost_model, **kwargs)

        # for distributed algorithm
        self.phi = np.zeros((self.M,1))
        self.psi = np.zeros((self.M,1))
        self.psi_last = np.zeros((self.M,1)) # used for exact diffusion
        self.name = 'agent '+str(kwargs.get('name', 'X'))

        self.combine_time = 0
        self.acc_time = 0

    def adapt(self, mu, ite, method='AVRG', style='Diffusion', **kwargs):
        self.psi_last = self.psi.copy()
        if style == 'Diffusion':
            self.psi = self.cost_model.w - mu * self.VR_option[method](ite, **kwargs)  
        elif style == 'EXTRA':
            self.psi = self.phi - mu * self.VR_option[method](ite, **kwargs)
        else:
            print ('Not support %s style' % style)
        
    def correct(self, ite, style='Diffusion'):
        if style =='Diffusion':
            self.phi = self.psi - self.psi_last + self.cost_model.w if ite!=0 else self.psi.copy()
        elif style == 'EXTRA':
            self.cost_model.w = self.psi - self.psi_last + self.phi if ite!=0 else self.psi.copy()
        else:
            print ('Not support %s style' % style)
            raise

    def combine(self, ite, style='Diffusion'):
        '''
        implemented by subclass
        '''
        pass

    def Performance(self, metric='MSD', w_=None):
        w = self.cost_model.w if w_ is None else w_

        if metric == 'MSD':
            return np.sum( (w - self.w_star)*(w - self.w_star) ) / self.norm_w_star
        elif metric == 'ER':
            return self.cost_model.func_value(w_ = w) - self.cost_model.func_value(w_ = self.w_star)
        else:
            print ('Unknown metric')
            return None

    def train(self, mu, max_ite, method = 'AVRG', dist_style = 'Diffusion', **kwargs):
        err = []
        # step-size need to adjust

        err_per_iter = kwargs.get('err_per_iter', 1)
        for ite in range(max_ite):
            if (ite % 1000) == 0 and self.name == 'agent 0':
                print ("Calculating iteration: ", ite)

            # adapt and correct
            acc_time =time.time()
            self.adapt(mu, ite, method, dist_style, **kwargs)
            self.correct(ite, dist_style)

            c_time = time.time()
            self.combine(ite, dist_style)
            self.combine_time += float(time.time() - c_time)
            self.acc_time += float(time.time() - acc_time)

            if ite % err_per_iter == 0:
                err_ = self.Performance(metric=kwargs.get('metric','MSD'))
                err.append(err_)

        print ('The total adapt-correct-combine time is %f' % self.acc_time)
        print ('The combine time is %f' % self.combine_time)
        print ('Percent of combine time is %.2f%%' % (self.combine_time / self.acc_time * 100))

        if self.name == 'agent 0':
            # print (err)
            # sio.savemat('err.mat', {'err':err} )
            plt.semilogy(err)
            plt.show()


class Multiprocess_VR_agent(multi_VR_agent_self):
    ''' 
        implementation of combinational step by multiprocess
    '''
    def __init__(self, X, y, w_star, cost_model, conns=None, **kwargs):
        '''
        kwargs usually need to proved the rho(regression coef.) for LR and name (int) for agent
        '''
        multi_VR_agent_self.__init__(self, X, y, w_star, cost_model, **kwargs)

        self.conns = conns
        self.neighbor = len(conns)

        self.name = 'agent '+str(kwargs.get('name', 'X'))

    def combine(self, ite, style='Diffusion'):
        '''
        communication through the Pipe of multiprocess
        '''
        if style not in ['Diffusion', 'EXTRA']:
            print ('Not support %s style' % style)
            raise

        # no nodes is connected to it
        if self.conns is None:
            if style == 'Diffusion':
                self.cost_model.w = self.phi.copy()
            elif style == 'EXTRA':
                self.phi = self.cost_model.w.copy()
            return

        if style == 'Diffusion': 
            for i in range(self.neighbor):
                # transform the w into row vector
                self.conns[i].send([self.phi.T.tolist(), self.neighbor])
            neigh_x = [self.phi.T.tolist()]
        elif style == 'EXTRA':
            for i in range(self.neighbor):
                # transform the w into row vector
                self.conns[i].send([self.cost_model.w.T.tolist(), self.neighbor])
            neigh_x = [self.cost_model.w.T.tolist()]
        weight_x = [0]

        for i in range(self.neighbor):
            value, neighbor = self.conns[i].recv()
            neigh_x.append(value)
            weight_x.append(1/(max(self.neighbor+1, neighbor+1))) # metropolis rule, 
                                                                  # +1 is because conns didn't count self

        weight_x[0] = 1 - np.sum(weight_x)
        neigh_x = np.asarray(neigh_x)

        if self.name == 'agent 0' and ite == 0:
            print ('weight_x shape: ', len(weight_x))
            print ('neigh_x', np.squeeze(neigh_x).shape)

        if style == 'Diffusion':
            self.cost_model.w = np.average(np.squeeze(neigh_x), weights=weight_x, axis = 0).reshape(-1,1)
        elif style == 'EXTRA':
            self.phi = np.average(np.squeeze(neigh_x), weights=weight_x, axis = 0).reshape(-1,1)


class ZMQ_VR_agent(multi_VR_agent_self):
    '''
        implementation of combinational step by zmq
        currently only support two-node under same LAN communication
    '''
    def __init__(self, X, y, w_star, cost_model, socket=None, **kwargs):
        '''
        kwargs usually need to proved the rho(regression coef.) for LR and name (int) for agent
        '''
        multi_VR_agent_self.__init__(self, X, y, w_star, cost_model, **kwargs)

        if socket is None:
            socket = []
        self.socket = socket
        self.neighbor = len(socket) ## only two nodes or no neighbor 

        # self.name = 'agent '+str(kwargs.get('name', 'X'))
        self.name = 'agent 0' #current call for print in all cmd

    def combine(self, ite, style='Difffusion'):
        '''
        communication through the Pipe of multiprocess
        '''
        if style not in ['Diffusion', 'EXTRA']:
            print ('Not support %s style' % style)
            raise

        if self.neighbor == 0:   # self update
            if style == 'Diffusion':
                self.cost_model.w = self.phi.copy()
            elif style == 'EXTRA':
                self.phi = self.cost_model.w.copy()
            return

        if style == 'Diffusion': 
            x = {'val': self.phi.tolist(), 'neighbor': self.neighbor}
        elif style == 'EXTRA': 
            x = {'val': self.cost_model.w.tolist(), 'neighbor': self.neighbor}

        self.socket[0].send_json(x)
        recv_x = self.socket[0].recv_json()

        # print
        # if ite == 10:
        #     print ('recived the shape of w: ',np.array(recv_x[u'val']).shape)
        #     print (np.array(recv_x[u'val'])[150:170])

        if style == 'Diffusion':
            self.cost_model.w = (self.phi + np.array(recv_x[u'val']))/(self.neighbor + 1)
        elif style == 'EXTRA':
            self.phi = (self.cost_model.w+np.array(recv_x[u'val']))/(self.neighbor + 1)