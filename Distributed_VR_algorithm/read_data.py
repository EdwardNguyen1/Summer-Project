from sklearn import datasets 
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os


def split_data(X, y, N_agent, even=True):
    '''
    split the data to N_agent chunks. Will remove the remainder data if all data can not split evenly
    '''
    if even:
        N_data_dist_prob = np.ones(N_agent) / N_agent
    else:
        N_data_dist_prob = np.random.rand(N_agent) + 0.2
        N_data_dist_prob = N_data_dist_prob / sum(N_data_dist_prob)


    N = X.shape[0]

    N_data_agent_list = N_data_dist_prob * N
    N_data_agent_list = N_data_agent_list.astype(int)
    N_data = np.sum(N_data_agent_list)

    X, y = X[0:N_data], y[0:N_data].reshape(-1, 1)

    N_data_boundary = np.hstack( (0, np.cumsum(N_data_agent_list)) )

    N_data_agent_mean = N_data // N_agent

    return X, y, N_data_boundary, N_data_agent_mean

def read_cov(**kwargs):
    covtype=datasets.fetch_covtype(data_home='../data')
    # extract only two label data 
    mask = np.in1d(covtype.target, [1,2])
    covtype.data = covtype.data[mask]
    covtype.target = covtype.target[mask]
    # preprocessing
    std = sklearn.preprocessing.StandardScaler()
    norm = sklearn.preprocessing.Normalizer()
    covtype.data = norm.fit_transform(covtype.data)
    X_train, X_test, y_train, y_test = \
        train_test_split(covtype.data, covtype.target, test_size=kwargs.get('test_size', 0.6))
    y_train[y_train==2] = -1
    y_test[y_test==2] = -1

    return X_train, y_train[:, np.newaxis]

def read_mnist(datatype ="binary", **kwargs):
    '''
    datatype: 'binary' or 'multiclass'; 'binary' is the default value
    mask_label: has type as list; default value [0, 1, 2, 3, 4]
    '''
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    # data_percentage = kwargs.get('data_percentage', 0.5)

    # replace it by your own path
    file_loc='./data/MNIST_data'

    if datatype == "binary":        
        ds = input_data.read_data_sets(file_loc, one_hot=False)

        X = ds.train.images
        y = ds.train.labels

        mask = np.in1d(y, [0,1])
        normalizer = sklearn.preprocessing.Normalizer()
        X= X[mask]
        X_train = normalizer.fit_transform(X)
        y = y[mask]
        y_train = (y-0.5)*2

        # y_train is N*1  vector
        return X_train, y_train[:, np.newaxis]

    elif datatype == 'multiclass':
        mask_label = kwargs.get('mask_label', [0, 1, 2, 3, 4])
       
        ds_oh = input_data.read_data_sets(file_loc, one_hot=True)

        X = ds_oh.train.images
        Y = ds_oh.train.labels
        y = np.argmax(Y, axis = 1)

        mask = np.in1d(y, mask_label)
        normalizer = sklearn.preprocessing.Normalizer()
        X = X[mask]
        X_train = normalizer.fit_transform(X)
        Y_train = Y[mask]

        # Y_train is N*C matrix with one_hot encoding
        return X_train, Y_train

def read_cifar():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    try:
        file_loc=os.path.abspath('C:/Users/biche/OneDrive/Documents/Python Scripts/random_reshuffle/real_data/cifar-10') 
        cifar_data = unpickle(os.path.join(file_loc,'data_batch_1'))
        cifar_data_test = unpickle(os.path.join(file_loc,'test_batch'))
    except:
        print ('did not find the cifar document')
        return None

    X_cifar_train = cifar_data[b'data']
    y_cifar_train = np.array(cifar_data[b'labels'])
    X_cifar_test = cifar_data_test[b'data']
    y_cifar_test = np.array(cifar_data_test[b'labels'])

    mask_train = np.in1d(y_cifar_train, [0,1])
    mask_test = np.in1d(y_cifar_test, [0,1])

    X_cifar_train = X_cifar_train[mask_train]
    y_cifar_train = y_cifar_train[mask_train]
    X_cifar_test = X_cifar_test[mask_test]
    y_cifar_test = y_cifar_test[mask_test]

    normalizer = sklearn.preprocessing.Normalizer()
    std = sklearn.preprocessing.StandardScaler()

    X_cifar_train = normalizer.fit_transform(X_cifar_train)
    X_cifar_test = normalizer.transform(X_cifar_test)
    
    y_cifar_train[y_cifar_train==0] = -1
    y_cifar_test[y_cifar_test==0] = -1

    return X_cifar_train, y_cifar_train[:, np.newaxis]