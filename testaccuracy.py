import gzip
import numpy as np
import ljmnistfuncs as lmf
def test(test_amount, weights, biases):
    print("loading MNIST test files")
    imgs = gzip.open('data/t10k-images-idx3-ubyte.gz', 'r')
    labs = gzip.open('data/t10k-labels-idx1-ubyte.gz', 'r')
    imgs.read(16) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    labs.read(8) #skip some data (this data can be useful but its more work to read it than to just hard code a couple numbers)
    imglst = []
    i = 0
    while i < 10000:
        buf = imgs.read(28 * 28 * 1)
        image = np.frombuffer(buf, dtype=np.uint8)
        labelbuf = labs.read(1)
        label = np.frombuffer(labelbuf, dtype=np.uint8)
        imglst.append((image, label))
        i+=1
    data = np.asarray(imglst)
    del imglst #die
    print("MNIST test files loaded")
    correct = 0
    item = 0
    while item < test_amount:
        desiredoutput = np.zeros(10)
        desiredoutput[data[item][1]] = 1
        
        acti, z = lmf.feedforwards(weights, biases, data[item][0])
        
        if data[item][1] == np.argmax(acti[-1]):
            correct +=1
        item +=1
    print("correctly classified {0} images out of {1}".format(correct, test_amount))
    return correct