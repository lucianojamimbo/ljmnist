import numpy as np
#sigmoid function, pretty simple
def sigmoid(z):
    return 1/(1+np.exp(np.negative(z)))
#deriv of sigmoid(x) MAKE SURE YOU PASS PRE-SIGMOID ACTIVATIONS INTO THIS NOT POST-SIGMOID
def sigmoidderiv(x):
    return (sigmoid(x)*(1-sigmoid(x)))
#create the required variables for our network
def init(sizes):
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    delta = list(np.zeros(len(sizes)-1)) #fill delta with 0s then convert to list
    return weights, biases, delta
#returns a list of all neuron activations, and z for pre-sigmoided activations
def feedforwards(w, b, a):
    activations = []
    z = []
    activations.append(a)
    for zw, zb in zip(w, b):
        bs = np.add(np.dot(zw, a), [item for sublist in zb for item in sublist]) 
        a = sigmoid(bs)
        activations.append(a)
        z.append(bs)
    return np.array(activations), np.array(z)
#basic cost function
def cost(y, a):
    return np.sum(np.power(np.subtract(y, a), 2))
#this one is important! this gets delta, the error for all biases and weights.
def getdelta(activations, desiredoutput, z, delta, weights):    
    delta[-1] = np.multiply(np.subtract(activations[-1], desiredoutput), sigmoidderiv(z[-1])) #error in the output layer
    i = 1 #start at one because error in output is already done
    while i < len(delta):
        delta[-i-1] = np.multiply(np.matmul(weights[-i].T, delta[-i]), sigmoidderiv(z[-i-1]))
        i+=1
#list to list of lists
def ltlol(lst):
    return [[itm] for itm in lst]
#get nabla_w_zero and nabla_b_zero
def nbnwzero(weights, biases, delta, sizes):
    acti, presig = feedforwards(weights, biases, np.zeros(sizes[0])) #run with a full zero input because we dont care about anything but the shapes
    do = np.zeros(sizes[-1]) #doesnt need to mean anything so its all zeros, we just want the shapes
    currentnablaw = []
    currentnablab = []
    getdelta(acti, do, presig, delta, weights)
    for a, d in zip(acti, delta):
        currentnablaw.append(np.transpose(np.matmul((ltlol(a)), np.transpose(ltlol(d)))))
    for d in delta:
        currentnablab.append(d)
    nabla_w_zero = np.multiply(currentnablaw, 0)
    nabla_b_zero = np.multiply(currentnablab, 0)
    return nabla_b_zero, nabla_w_zero