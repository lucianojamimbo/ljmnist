from loaddata import ld
from testaccuracy import test
import numpy as np
import ljmnistfuncs as lmf #import all the basic functions
from train import trainSGD
np.random.seed(0) #make the program behave the same way every time it is run so that its easier to track the effect of changes to the code



#init time
sizes = [784,16,10]
weights, biases, delta = lmf.init(sizes) #init the main variables we need
data = ld() #only loading to test the loading function works, this data is unused *currently*
np.random.shuffle(data) #randomise our data
nabla_b_zero, nabla_w_zero = lmf.nbnwzero(weights, biases, delta, sizes) #get nabla_w_zero and nabla_b_zero
#testing and training
test(10000, weights, biases)
weights, biases = trainSGD(5, 10, 0.09, data, nabla_w_zero, nabla_b_zero, sizes, delta, weights, biases)
test(10000, weights, biases)