import numpy as np
from loaddata import ld
#sigmoid function, pretty simple
def sigmoid(z):
    return 1/(1+np.exp(np.negative(z)))
#deriv of sigmoid(x) MAKE SURE YOU PASS PRE-SIGMOID ACTIVATIONS INTO THIS NOT POST-SIGMOID
def sigmoidderiv(x):
    return (sigmoid(x)*(1-sigmoid(x)))
#create the required variables for our network
def init(sizes):
    global delta
    global biases
    global weights
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    delta = list(np.zeros(len(sizes)-1)) #fill delta with 0s then convert to list
#returns a list of all neuron activations, and z for pre-sigmoided activations
def feedforwards(w, b, a):
    activations = []
    z = []
    activations.append(a)
    for zw, zb in zip(w, b):
        bs = np.add(np.dot(zw, a), [item for sublist in zb for item in sublist]) 
        a = sigmoid(np.add(np.dot(zw, a), [item for sublist in zb for item in sublist]))
        activations.append(a)
        z.append(bs)
    activations = np.array(activations)
    z = np.array(z)
    return activations, z
def cost(y, a):
    return np.sum(np.power(np.subtract(y, a), 2))

def getdelta(activations, desiredoutput, z):    
    delta[-1] = np.multiply(np.subtract(activations[-1], desiredoutput), sigmoidderiv(z[-1])) #error in the output layer
    i = 1 #start at one because error in output is already done
    while i < len(delta):
        delta[-i-1] = np.multiply(np.matmul(weights[-i].T, delta[-i]), sigmoidderiv(z[-i-1]))
        i+=1


#make sure the functions work:
do = [0.2,0.2] #a variable required for testing cost()
init([2,2,2])
acti, presig = feedforwards(weights, biases, [0.1,0.2])
data = ld()
getdelta(acti, do, presig)
print("cost", cost(do, acti[-1]))
print("first entry in data", data[0])
print("activations", acti)
print("pre-sigmoid activations", presig)
print("delta", delta)