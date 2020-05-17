from loaddata import ld
from testaccuracy import test
import numpy as np
import ljmnistfuncs as lmf #import all the basic functions
from train import trainSGD
import time
np.random.seed(0) #make the program behave the same way every time it is run so that its easier to track the effect of changes to the code
sizes = [784,30,10] #this decides the shape of our network.
weights, biases, delta = lmf.init(sizes) #init the main variables we need
data = ld() #only loading to test the loading function works, this data is unused *currently*
np.random.shuffle(data) #randomise our data
nabla_b_zero, nabla_w_zero = lmf.nbnwzero(weights, biases, delta, sizes) #get nabla_w_zero and nabla_b_zero
start = time.time()
#epoch_amount, batch_size, eta, nabla_w_zero, nabla_b_zero, sizes, delta, weights, biases
weights, biases = trainSGD(30, 10, 0.003, data, nabla_w_zero, nabla_b_zero, sizes, delta, weights, biases) #training time
test(10000, weights, biases) #test time
end = time.time()
print("time taken (s):", end-start)
#old best: 8283/10000 (seed 0, batch size 10, learning rate 0.1, 30 epochs, [784,30,10])
#new best: 8460/10000 (seed 0, batch size 10, learning rate 0.15, 30 epochs, [784,30,10])