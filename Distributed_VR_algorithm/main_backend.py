from flask import Flask, render_template, jsonify, request, Response
app = Flask(__name__)

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from read_data import *
from VR_algorithm import *
from cost_func import soft_max
from scipy.misc import imresize
import zmq
from multiprocessing import Process

@app.route("/", methods=['GET'])
def index():
    return render_template("topology.html")

@app.route("/topo/<int:N_node>", methods=['GET'])
def get_top(N_node):
    G = generate_topology(N_node, prob=0.5)
    return jsonify({'node': G.node, 'edge': G.edge})

@app.route("/image", methods=['GET'])
def get_image():
    # img = plt.imread('IMG_test.jpg')
    # imglist = np.dstack( [img, np.ones(img.shape[:2])] )

    # return jsonify({'img_data': imglist.ravel().tolist(),
    #                 'img_shape': img.shape})
    return jsonify({'img_addr': '/static/visual_W_5000.jpg'})

def W_to_img(w, **kwargs):
    # save W into images (only supoort 10-class mnist problem now)
    W = w.reshape(28*28, 10, order='F')
    padding = 1
    trans_W = np.ones((28*2+1*padding, 28*5+4*padding))*np.min(W) # 2*5 range middle has padding

    for i in range(5):
        x_start = i*28+i*padding
        trans_W[:28, x_start:x_start+28] = W[:,i].reshape(28,28)
        trans_W[28+padding:, x_start:x_start+28] = W[:,i+5].reshape(28,28)

    # preprocessing trans_W
    ratio = trans_W.shape[1]/float(trans_W.shape[0])
    final_W = imresize(trans_W, (200,int(200*ratio)), interp='bilinear')

    save_file = os.path.join('./static', 'visual_W_%s.jpg'%sys.argv[1])
    plt.imsave(save_file,final_W)

# global variable shared between functions
vr_alg = None
X,Y = None, None
sockets = []

@app.route("/connect", methods=['POST'])
def get_connections():
    ip_addr = request.json['ip']
    cs = request.json['server_client']
    global sockets
    if cs == "server":
        try:
            context = zmq.Context()
            s = context.socket(zmq.PAIR)
            s.bind("tcp://"+ip_addr)
            sockets.append(s)
            return Response(None)
        except:
            print ("Error happened when connected server")
            s.close()
            return Response(status=500)
    elif cs == "client":
        try:
            context = zmq.Context()
            s = context.socket(zmq.PAIR)
            s.connect("tcp://"+ip_addr)
            sockets.append(s)
            return Response(None)
        except:
            print ("Error happened when connected server")
            s.close()
            return Response(status=500)
    else:
        return Response(status=500)

@app.route("/disconnect", methods=['GET'])
def disconnect():
    global sockets
    [s.close() for s in sockets]
    sockets = []
    return Response(None)

@app.route('/get_data', methods=['POST'])
def get_data():
    tmp_mask=request.json['mask']
    mask = [int(i) for i in tmp_mask if tmp_mask[i]]
    # print (tmp_mask)

    global vr_alg, X, Y, socket
    X,Y = read_mnist(datatype="multiclass", mask_label=mask)

    # for computing cost function fast
    X = X[:int(X.shape[0]*0.2)]
    Y = Y[:int(Y.shape[0]*0.2)]
    
    return Response(None)


@app.route('/run_alg', methods=['POST'])
def run_alg( **kwargs):
    if X is None or Y is None:
        return Response(response={'message': "No data loaded yet"}, status=500)

    mu = float(request.json['mu'])
    method = request.json['method']
    start_ite = int(request.json['ite'])
    dist_style = request.json['dist_style']
    iter_per_call = int(request.json['iter_per_call'])
    
    global vr_alg
    if start_ite == 0:
        vr_alg = ZMQ_VR_agent(X,Y, np.random.randn(28*28*10,1), soft_max, socket=sockets, rho = 1e-4)

    W_to_img(vr_alg.cost_model.w)

    for ite in range(start_ite, start_ite+int(iter_per_call)):
        # grad_modifed = vr_alg.VR_option[method](ite, **kwargs)
        # vr_alg.cost_model._update_w(grad_modifed, mu)

        vr_alg.adapt(mu, ite, method, dist_style, **kwargs)
        vr_alg.correct(ite, dist_style)
        vr_alg.combine(ite, dist_style)

    if start_ite % (2*iter_per_call) == 0:
        cost_value = vr_alg.cost_model.func_value()
    else:
        cost_value = 'skipped'

    return jsonify({'cost_value': cost_value, 'running_port': sys.argv[1]})

@app.route('/rest_alg', methods=['GET'])
def reset_alg():
    if vr_alg is None:
        W_to_img(np.zeros((10*28*28,1)) )
        return jsonify({'running_port': sys.argv[1]})

    vr_alg.reset()
    W_to_img(vr_alg.cost_model.w)
    return jsonify({'running_port': sys.argv[1]})

if __name__ == '__main__':
    #p = Process(target=run_alg, args=(vr_alg, 3e5, 0.02,'SVRG'), kwargs={'using_sgd':1, 'replace': True})
    # p.start()

    app.run(debug=False, port = sys.argv[1])

    