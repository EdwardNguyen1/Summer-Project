import numpy as np

def gen_gaussian_data(N,M=5,sigma=0.1):
    X = np.random.randn(N,M)
    w_0 = np.random.randn(M,1)
    noise = sigma*np.random.randn(N,1)
    y = X.dot(w_0) + noise
    w_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return X, y, w_star

def gen_lr_data(N,M=5):
    X1 = np.random.randn(int(N/2),M)+1
    X2 = np.random.randn(int(N/2+0.5),M)-1
    y1 = np.ones( (int(N/2), 1))
    y2 = -1*np.ones( (int(N/2), 1))
    shuffle = np.random.permutation(N)
    X = np.vstack([X1,X2])[shuffle,:]
    y = np.vstack([y1,y2])[shuffle]
    
#     X = np.random.randn(N,M)
#     w_0 = np.random.randn(M,1)
#     prob = np.exp(X.dot(w_0)) / (1+np.exp(X.dot(w_0)))
#     y = np.sign( np.random.rand(N,1) - prob )
    return X, y#, w_0

class least_square():
    '''
        solving the min_w \|Xw-y\|^2/2/N LMS problem
        X---------N*M (# of instance * # dimension of problem)
        y---------N*1 (# of instance * 1)
        w---------M*1 (# dimension of problem * 1)
        The full gradient is 
                X'*(X*w-y)/N
    '''
    def __init__(self, X, y, w=None, **kwargs):
        self.X, self.y = X, y
        self.N = X.shape[0]
        self.M = X.shape[1]
        if w is None:
            self.w = np.zeros( (self.M, 1) )
        else:
            assert w.shape == (self.M, 1), \
                    "the shape of given w should be M*1 (the same dimension of size of second dimension of X"
            self.w = w
    
    def full_gradient(self, w_=None):
        w = self.w if w_ is None else w_
        
        return self.X.T.dot(self.X.dot(w)-self.y)/self.N
        
    def partial_gradient(self, index=None, w_=None):
        '''
            return a partial gradient by index
            if index is none, the function return the gradient based on one (uniformly) random realization
            if index is not none, the function will return the gradient based on selected index realization
            index can be int or the list of int
            index \in [0, N-1]
        '''
        w = self.w if w_ is None else w_
        
        if index == None:
            index = np.random.randint(self.N, size=1)
        
        # check the format
        if isinstance(index, int) or isinstance(index, list) or isinstance(index, np.ndarray):
            pass
        else:
            raise ValueError("index should be either int or list of int between [0,N-1]")
        
        if isinstance(index, int):
            error_ = self.X[[index], :].dot(w) - self.y[index]
            return self.X.T[:,[index]]*error_
        else:
            error_ = self.X[index, :].dot(w) - self.y[index]
            return self.X.T[:,index].dot(error_)/float(len(index))
            
    def gradient_by_data(self, w_=None):
        '''
            return N*M matrix, each row (1*M) is the gradient at w_ defined by one data
        '''
        w = self.w if w_ is None else w_
            
        return (self.X.dot(w)-self.y)*self.X
    
    def func_value(self, w_=None):
        w = self.w if w_ == None else w_
        
        return np.sum(np.square(self.X.dot(w)-self.y))/2/self.N
    
    def Hessian(self, w_=None):
        return self.X.T.dot(self.X)/self.N
    
    def _update_w(self, gradient, mu=0.01):
        self.w = self.w - mu*gradient
    
    def _reset_w(self, rand=False):
        if rand:
            self.w = np.random.randn(self.M, 1)*0.01
        else:
            self.w = np.zeros( (self.M, 1) ) 


class logistic_regression():
    '''
        solving the min_w mean(\ln[1+exp(-yX*w)]) + rho/2\|w\|^2 LR problem
        X---------N*M (# of instance * # dimension of problem)
        y---------N*1 (# of instance * 1) (should be +/- 1)
        w---------M*1 (# dimension of problem * 1)
        The full gradient is 
            rho*w -  mean(exp(-yX*w)/( 1 + exp(-yX*w) )yX)
    '''
    def __init__(self, X, y, **kwargs):
        self.X, self.y = X, y
        xx = {i[0] for i in y}
        assert xx=={1.0,-1.0},  "y should only contain +1, -1"
        self.N = X.shape[0]
        self.M = X.shape[1]

        self.w = kwargs.get('w', np.zeros( (self.M, 1) ))
        assert self.w.shape == (self.M, 1), \
                    "the shape of given w should be 1xM (the same dimension of size of second dimension of X"
        self.rho = kwargs.get('rho', 0.01)
    
    def full_gradient(self, w_=None):

        w = self.w if w_ == None else w_
        
        exp_val = np.exp(-self.y*self.X.dot(w))
        return self.rho*w - np.mean( exp_val/(1+exp_val)*self.y*self.X, axis=0)[:,np.newaxis]
    
    def partial_gradient(self, index, w_=None):
        '''
            return a partial gradient by index
            if index is none, the function return the gradient based on one (uniformly) random realization
            if index is not none, the function will return the gradient based on selected index realization
            index can be int or the list of int
            index \in [0, N-1]
        '''
        w = self.w if w_ is None else w_
        
        # if index == None:
        #     index = np.random.randint(self.N, size=1)
        # # check the format
        # if isinstance(index, int) or isinstance(index, list) or isinstance(index, np.ndarray):
        #     pass
        # else:
        #     raise ValueError("index should be either int or list of int between [0,N-1]")
        
        if not isinstance(index, list) and not isinstance(index, np.ndarray): 
            exp_val = np.exp(-self.y[index]*self.X[index,:].dot(w))
            return self.rho*w - (exp_val/(1+exp_val)*self.y[index]*self.X[index,:])[:,np.newaxis]
        else:
            exp_val = np.exp(-self.y[index]*self.X[index,:].dot(w))
            return self.rho*w - np.mean(exp_val/(1+exp_val)*self.y[index]*self.X[index,:],axis=0)[:,np.newaxis]
            
    
    def gradient_by_data(self, w_=None):
        '''
            return N*M matrix, each row (1*M) is the gradient at w_ defined by one data
        '''
        w = self.w if w_ is None else w_
        
        exp_val = np.exp(-self.y*self.X.dot(w))
        return self.rho*w.T -  (exp_val/(1+exp_val)*self.y*self.X)
    
    def Hessian(self, w_=None):
        if w_ is None:
            w = self.w
        else:
            w = w_
            
        exp_val = np.exp(-self.y*self.X.dot(w))
        return self.rho*np.eye(self.M) + self.X.T.dot(exp_val/(1+exp_val)/(1+exp_val)*self.X)/self.N

    def func_value(self, w_=None):
        w = self.w if w_ is None else w_
        
        return np.mean(np.log(1+np.exp(-self.X.dot(w)*self.y))) + self.rho/2 * np.sum(np.square(w))

    def _update_w(self, gradient, mu=0.01):
        self.w -= mu*gradient
    
    def _reset_w(self, rand=False):
        if rand:
            self.w = np.random.randn(self.M, 1)*0.01
        else:
            self.w = np.zeros( (self.M, 1) ) 

class soft_max():
    '''
        solving the min_W mean(   - log ( y_n.T exp(f_n) / sum( exp(f_n) ) ) )  + rho/2\|W\|^F soft_max problem, where f = W * x
        X --------- N * D (# of instance * # dimension of problem)
        y --------- N * C 
        W --------- D * C (# dimension of data * # classes)
    '''
    def __init__(self, X, Y, **kwargs):
        self.X, self.Y = X, Y
        # xx = {i[0] for i in y}
        # assert xx=={0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},  "y should only contain 0, 1, ..., 9"
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.C = Y.shape[1]
        self.M = self.D * self.C

        self.w = kwargs.get('w', np.zeros((self.M, 1)))
        assert self.w.shape == (self.M, 1), \
                    "the shape of given w should be M (M = D * C)"
        
        # check whether Y is one-hot coding
        assert self.Y.ndim == 2, "Y shold be one-hot coding"
        assert all(np.unique(Y.flatten()) == np.array([0.,1.])),  "y should only contain 0, 1 (one-hot coding)"
        assert set(np.sum(self.Y, axis=1)) == {1.0}, "Y shold be one-hot coding"        

        self.rho = kwargs.get('rho', 0.01)
    
    def partial_gradient(self, index, w_=None):
        '''
            return a partial gradient by index
            if index is none, the function return the gradient based on one (uniformly) random realization
            if index is not none, the function will return the gradient based on selected index realization
            index can be int or the list of int
            index \in [0, N-1]
        '''
        w = self.w if w_ is None else w_
        W = w.reshape(self.D, self.C, order='F')
        
        # if index == None:
        #     index = np.random.randint(self.N, size=1)
        # # check the format
        # if isinstance(index, int) or isinstance(index, list) or isinstance(index, np.ndarray):
        #     pass
        # else
        #     raise ValueError("index should be either int or list of int between [0,N-1]")
        
        if not isinstance(index, list) and not isinstance(index, np.ndarray): 
            y_n = self.Y[index,:].reshape(-1, 1)
            x_n = self.X[index,:].reshape(-1, 1)
            f_n = W.T.dot(x_n)
            prob = np.exp(f_n) / np.sum( np.exp(f_n) )
            partial_grad_mat = (prob - y_n).T * x_n + self.rho * W
        else:
            Y_n = self.Y[index,:]
            X_n = self.X[index,:]
            F_n = X_n.dot(W)
            prob = np.exp(F_n) / np.sum( np.exp(F_n), axis=1, keepdims=True)
            partial_grad_mat = X_n.T.dot(prob - Y_n)/len(index) + self.rho * W
        
        return partial_grad_mat.reshape(-1, 1, order = 'F')  


    def full_gradient(self, w_=None):
        w = self.w if w_ is None else w_
        W = w.reshape(self.D, self.C, order='F')

        F = self.X.dot(W)
        prob = np.exp(F) / np.sum( np.exp(F), axis=1, keepdims=True)
        partial_grad_mat = self.X.T.dot(prob - self.Y)/self.N + self.rho * W
        return partial_grad_mat.reshape(-1, 1, order = 'F')  

    # def partial_func_value(self, index, w_=None):
    #     w = self.w if w_ is None else w_
    #     W = w.reshape(self.D, self.C, order='F')

        
    #   y_n = self.Y[index,:].reshape(-1, 1)
    #   x_n = self.X[index,:].reshape(-1, 1)
    #   f_n = W.T.dot(x_n)
    #   prob = np.exp(f_n) / np.sum( np.exp(f_n) )
    #   f_value -= np.log( y_n.T.dot(prob) ) / self.N
        
    #     return f_value + (self.rho / 2) * np.sum( W *  W)


    def func_value(self, w_=None):
        w = self.w if w_ is None else w_
        W = w.reshape(self.D, self.C, order='F')
        
        F = self.X.dot(W)
        prob = np.exp(F) / np.sum( np.exp(F), axis=1, keepdims=True )
        f_value = -np.mean( np.log(np.sum(prob*self.Y, axis=1)) )
        
        return float(f_value) + (self.rho / 2) * np.sum( W *  W)
            
    
    # def gradient_by_data(self, w_=None):
    #     '''
    #         return N*M matrix, each row (1*M) is the gradient at w_ defined by one data
    #     '''
    #     w = self.w if w_ is None else w_
        
    #     exp_val = np.exp(-self.y*self.X.dot(w))
    #     return self.rho*w.T -  (exp_val/(1+exp_val)*self.y*self.X)
    
    # def Hessian(self, w_=None):
    #     if w_ is None:
    #         w = self.w
    #     else:
    #         w = w_
            
    #     exp_val = np.exp(-self.y*self.X.dot(w))
    #     return self.rho*np.eye(self.M) + self.X.T.dot(exp_val/(1+exp_val)/(1+exp_val)*self.X)/self.N

    def _update_w(self, gradient, mu=0.01):
        self.w -= mu*gradient
    
    def _reset_w(self, rand=False):
        if rand:
            self.w = np.random.randn(self.M, 1)*0.01
        else:
            self.w = np.zeros( (self.M, 1) ) 

    def GD(self, maxite = 2000, mu = 0.1):
        for i in range(maxite):
            if i % 100 == 0:
                print ('iteration:', i)
            self.w -= mu*self.full_gradient()
        return self.w

















