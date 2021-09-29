from keras.datasets.mnist import load_data
from DNNPretrained import DNNPretrained
from mainRetrain import getDatasets, getBinaryClass
import numpy as np
import os
import sys


def main():
    """
    Main method used to write data sets and weights to files for improvement methods
    """
    # load the DNNs
    dnn = DNNPretrained("modelBasic")
    dnnPR = DNNPretrained("modelBasic")
    dnnCR = DNNPretrained("modelCompRetrain")

    # load training data
    advExmpls, advClass , origImg, origClass = getDatasets(False)

    advOrigImg = np.vstack((advExmpls,origImg))
    advOrigClass = np.vstack((advClass, origClass))

    # perform the partial retraining
    dnnPR.trainAdvExmpls(advOrigImg, advOrigClass)

    # write the weights of the DNNs used
    dnn.writeWeights("output\\weights\\afterImpr\\weightsIP.csv")
    # dnnPR.writeWeights("output\\weights\\afterImpr\\weightsPR.csv")
    # dnnCR.writeWeights("output\\weights\\afterImpr\\weightsCR.csv")

    # write the test data
    writeTestDataAfterImpr(dnn, dnnPR, dnnCR)

def writeTestDataAfterImpr(dnn, dnnPR, dnnCR):
    """
    Method used to write test data that is correctly classified by all improvement approaches
    :param dnn:         The DNN used when applyin the perturbation
    :param dnnPR:       The partial retrained DNN
    :param dnnCR:       The completely retrained DNN
    """
    # load the test data
    _, _, origImg, origClass, _, _ = getDatasets(True)

    # get the perturbation and perturb the images
    p, q = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbationMinDist.csv", delimiter=","))
    origImgPerturbed = np.multiply(origImg, p)
    origImgPerturbed = np.add(origImgPerturbed, q)

    # get the images that are classified correctly by every improvement method
    filter1 = dnnPR.testAccuracy(origImg, origClass, returnCorrectIndex=True)
    filter2 = dnnCR.testAccuracy(origImg, origClass, returnCorrectIndex=True)
    filter3 = dnn.testAccuracy(origImgPerturbed, origClass, returnCorrectIndex=True)

    # get the images that are classified correctly by all approaches
    filter = np.logical_and(filter1, filter2)
    filter = np.logical_and(filter, filter3)

    # set the seed to get consistent results
    np.random.seed(0)
    # randomly reorder the images to get images of different digits
    order = np.array(range(origImg[filter].shape[0]))
    np.random.shuffle(order)

    # randomly select 100 correctly classified images
    X = origImg[filter][order][:100]
    y = np.argmax(origClass[filter][order][:100],axis=1)
    y = np.array(y, dtype=int)

    # set the directory the data set will be added to
    dir = "output\\testdataPerformance\\afterImpr\\"
    files = [("images.csv", X), ("classifications.csv", y)]
    # write the data sets(images and classifications) to csv files
    for file in files:
        filename = dir + file[0]
        oldFile = os.path.join(sys.path[0], filename)

        if os.path.exists(oldFile):
            os.remove(oldFile)

        with open(filename, "a") as f:
            f.write(str(file[1].shape[0]) + "\n")
            np.savetxt(f, file[1], fmt='%s', delimiter=",")

if __name__ == '__main__':
    main()
