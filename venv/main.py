from keras.datasets.mnist import load_data
from DNN import DNN
from mainRetrain import getDatasets, getBinaryClass
import numpy as np


def main():
    """
    Main method used to train DNNs for the 5 DNNs and save their weights to a csv file
    """
    # load in the MNIST data
    (XTrain, yTrain), (XTest, yTest) = load_data(path="mnist.npz")

    # Preprocess the data (these are NumPy arrays)
    XTrain = XTrain.reshape(60000, 784).astype("float32") / 255
    XTest = XTest.reshape(10000, 784).astype("float32") / 255

    # convert the classifications into lists of binary dummies
    yTrainBin = getBinaryClass(yTrain)
    yTestBin = getBinaryClass(yTest)

    # Reserve 10,000 samples for validation
    XVal = XTrain[-10000:]
    yVal = yTrainBin[-10000:]
    XTrain = XTrain[:-10000]
    yTrainBin = yTrainBin[:-10000]

    # create a list containing the 5 DNNs' architectures and their respective learning rates
    architectures = [
        ((8,8,8),0.001),
        ((8,8,8,8,8), 0.001),
        ((20,10,8,8), 0.0015),
        ((20,10,8,8,8), 0.001),
        ((20,20,10,10,10), 0.0025)
    ]
    adversarial = False

    # # for the complete retraining use this list
    # architectures = [((8,8,8),0.001)]
    # adversarial = True

    for architecture in architectures:
        # get the training data
        archX = np.copy(XTrain)
        archY = np.copy(yTrainBin)
        if adversarial:
            # add the adversarial examples to the training data
            advExmpls, advClass , origImg, origClass = getDatasets(False)
            archX = np.vstack((archX, advExmpls))
            archY = np.vstack((archY, advClass))

        # initialize and train the DNN
        dnn = DNN(archX, archY, XTest, yTestBin, XVal, yVal, architecture[0], architecture[1])

        if adversarial:
            # # save the model if necessary
            # dnn.save("modelCompRetrain")

            # write the weights to a csv file
            dnn.writeWeights("weightsCompRetrain.csv")
        else:
            # # save the model if necessary
            # dnn.save("modelBasic")

            # write the weigths and test data to a csv file
            dnn.writeWeights("weights.csv")
            dnn.writeTestdataPerformance()


if __name__ == "__main__":
    main()
