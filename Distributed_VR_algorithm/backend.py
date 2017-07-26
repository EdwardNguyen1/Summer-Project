from flask import Flask, render_template, jsonify, request, Response
import sys
import zmq
import time
app = Flask(__name__)

vr_alg = None
X,Y = None, None
socketlist = []
iplist = []
port = []
count = []
connectionDict = {}
@app.route("/", methods=['GET'])
def index():
    return render_template("getIP.html")

@app.route("/buildtable", methods=['GET'])
def buildtable(): 
    print ("Sending ip...")
    return jsonify({'iplist': iplist})

@app.route("/sendIP", methods=['POST'])
def retrieveIP():
    ip_addr = request.json['retrievedIP']
    iplist.append(ip_addr)
    return Response(None)

@app.route("/connect", methods=['GET'])
def connect():
    portNumber = 5000
    counter = 1
    iplistreversed = iplist[::-1]
    for x in range(len(iplist)):
        port.append(portNumber)
        portNumber += 1
    for y in range(len(iplist)):
        count.append(counter)
        counter += 1
    combinedList = zip(iplist, iplistreversed, port)
    for k in range(len(combinedList)):
        connectionDict[k+1]= combinedList[k]
    print('Success!')
    for x in iplist:
        print (x)
    return Response(None)
    
if __name__ == '__main__':
    app.run(host="192.168.0.15", debug=False, port = sys.argv[1])
