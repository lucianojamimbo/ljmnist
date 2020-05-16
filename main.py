from loaddata import ld
from testaccuracy import test
import numpy as np
import ljmnistfuncs as lmf #import all the basic functions
np.random.seed(0) #make the program behave the same way every time it is run so that its easier to track the effect of changes to the code











def trainSGD(epochs, batch_size, eta):
    global weights
    global biases
    epoch = 0
    while epoch < epochs:
        i = 0
        dataiter = 0
        while i < 60000/batch_size:
            #make nabla_w and nabla_b full of zeros in the correct shape:
            nabla_w = nabla_w_zero
            nabla_b = nabla_b_zero
            #get a batch
            btch = 0
            batch = []
            while btch < batch_size:
                batch.append(data[dataiter])
                btch+=1
                dataiter+=1
            #do stuff
            for item in batch:
                desiredoutput = np.zeros(sizes[-1])
                desiredoutput[int(item[1])] = 1
                currentnablaw = []
                currentnablab = []
                acti, presig = lmf.feedforwards(weights, biases, item[0])
                lmf.getdelta(acti, desiredoutput, presig, delta, weights) #there is an issue with this somewhere
                for a, d in zip(acti, delta):
                    currentnablaw.append(np.transpose(np.matmul((lmf.ltlol(a)), np.transpose(lmf.ltlol(d)))))
                for d in delta:
                    currentnablab.append(d)
                nabla_w = np.add(nabla_w, currentnablaw)
                nabla_b = np.add(nabla_b, currentnablab)
            #below changes the weights and biases according to nabla_b and nabla_w
            weights -= np.dot((eta/batch_size), nabla_w)
            nbi = 0
            for nb in nabla_b:
                biases[nbi] -= lmf.ltlol(np.dot((eta/batch_size), nb))
                nbi+=1
            i+=1
        print("epoch {0} complete".format(epoch+1))
        epoch +=1












sizes = [784,16,10]
weights, biases, delta = lmf.init(sizes) #init the main variables we need
data = ld() #only loading to test the loading function works, this data is unused *currently*
np.random.shuffle(data) #randomise our data
#all this is required to get the shapes of nabla_w and nabla_b (bit of a hassle but no way around this)
do = [0,0,0,0,0,0,0,0,0,0]
acti, presig = lmf.feedforwards(weights, biases, np.zeros(sizes[0])) #run with a full zero input because we dont care about anything but the shapes
desiredoutput = np.zeros(sizes[-1]) #doesnt need to mean anything so its all zeros, we just want the shapes
currentnablaw = []
currentnablab = []
lmf.getdelta(acti, do, presig, delta, weights)
for a, d in zip(acti, delta):
    currentnablaw.append(np.transpose(np.matmul((lmf.ltlol(a)), np.transpose(lmf.ltlol(d)))))
for d in delta:
    currentnablab.append(d)
nabla_w_zero = np.multiply(currentnablaw, 0)
nabla_b_zero = np.multiply(currentnablab, 0)

#testing and training
test(10000, weights, biases)
trainSGD(5, 10, 0.09)
test(10000, weights, biases)