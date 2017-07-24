from flask import Flask, render_template, jsonify, request, Response
import sys
import zmq
app = Flask(__name__)
#check code with Kun
vr_alg = None
X,Y = None, None
socketlist = []
iplist = []
port = []
@app.route("/", methods=['GET'])
def index():
    return render_template("getIP.html")

@app.route("/sendIP", methods=['POST'])
def retrieveIP():
    ip_addr = request.json['retrievedIP']
    iplist.append(ip_addr)
    return Response(None)

@app.route("/connect", methods=['GET'])
def connect():
    count = 0
    count1 = 0
    portNumber = 5000
    port.append(portNumber)
    while count < len(iplist):
        context = zmq.Context()
        s = context.socket(zmq.PAIR)
        s.bind("tcp://"+iplist[count]+":"+port[count])
        socketlist.append(s)
        portNumber += 1
        port.append(portNumber)
        count += 1
    while count1 < (len(iplist)-1):
        context = zmq.Context()
        s = context.socket(zmq.PAIR)
        s.connect("tcp://"+iplist[count+1]+":"port[count1])
        socketlist.append(s)
        count1 +=1
if __name__ == '__main__':
    app.run(debug=False, port = sys.argv[1])
