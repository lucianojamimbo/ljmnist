from loaddata import ld
import numpy as np
import ljmnistfuncs as lmf #import all the basic functions

#make sure the functions work:
do = [0.2,0.2] #a variable required for testing cost()
weights, biases, delta = lmf.init([2,2,2])
acti, presig = lmf.feedforwards(weights, biases, [0.1,0.2])
data = ld()
lmf.getdelta(acti, do, presig, delta, weights)
np.random.shuffle(data) #randomise our data