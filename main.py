from loaddata import ld
import numpy as np
import ljmnistfuncs as lmf #import all the basic functions



#lets attempt to get a working training algorithm
def trainSGD(epochs, batch_size):
    epoch = 0
    while epoch < epochs:
        i = 0
        dataiter = 0
        while i < 60000:
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
                
            
            
            
            
            
            
            i+=1
        epoch +=1


do = [0.2,0.2] #currently not set based on the train data, just set for test
sizes = [2,2,2] #not set for use, just for testing
weights, biases, delta = lmf.init(sizes) #init the main variables we need
data = ld() #only loading right now so i can test
np.random.shuffle(data) #randomise our data
#all this is required to get the shapes of nabla_w and nabla_b (bit of a hassle but no way around this)
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

#test
trainSGD(1, 6)