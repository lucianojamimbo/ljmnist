from loaddata import ld
import ljmnistfuncs #import all the basic functions

#make sure the functions work:
do = [0.2,0.2] #a variable required for testing cost()
weights, biases, delta = ljmnistfuncs.init([2,2,2])
acti, presig = ljmnistfuncs.feedforwards(weights, biases, [0.1,0.2])
data = ld()
ljmnistfuncs.getdelta(acti, do, presig, delta, weights)
print("cost", ljmnistfuncs.cost(do, acti[-1]))
print("first entry in data", data[0])
print("activations", acti)
print("pre-sigmoid activations", presig)
print("delta", delta)