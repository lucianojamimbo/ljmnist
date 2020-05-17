from testaccuracy import test
import numpy as np
import ljmnistfuncs as lmf
import matplotlib.pyplot as plt
import copy
def trainSGD(epochs, batch_size, eta, data, nabla_w_zero, nabla_b_zero, sizes, delta, weights, biases):
    graph = []
    epoch = 0
    np.random.shuffle(data)
    while epoch < epochs:
        i = 0
        dataiter = 0
        while i < 60000/batch_size:
            #make nabla_w and nabla_b full of zeros in the correct shape:
            nabla_w = copy.deepcopy(nabla_w_zero)
            nabla_b = copy.deepcopy(nabla_b_zero)
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
            nabla_w = np.divide(nabla_w, batch_size)
            nabla_b = np.divide(nabla_b, batch_size)
            #below changes the weights and biases according to nabla_b and nabla_w
            weights -= np.multiply(np.divide(eta, batch_size), nabla_w)
            nbi = 0
            for nb in nabla_b:
                biases[nbi] -= lmf.ltlol(np.multiply(np.divide(eta, batch_size), nb))
                nbi+=1
            i+=1
        print("epoch {0} complete".format(epoch+1))
        graph.append(test(1000, weights, biases))
        epoch +=1
    plt.plot(graph)
    return weights, biases