import keras
import numpy as np
import os
import sys
from keras.models import load_model
from tensorflow.random import set_seed
from random import seed


class DNNPretrained:
    """
    Class used to model a DNN that loads in pretrained weights
    and biases from a directory
    """

    def __init__(self, filename):
        """
        Initializes the DNN by loading weights and biases from the given files
        :param filename:    Name of the directory containing the model
        """
        # set the seeds to get consistent results
        set_seed(0)
        seed(0)
        # load the weights and biases from the file
        self.DNN = load_model("models\\" + filename)

    def trainAdvExmpls(self, advExmpls, classifications):
        """
        Method used to train the DNN on a set of adversarial examples
        :param advExmpls:           The adversarial examples used as the training data
        :param classifications:     The correct classification of the adversarial examples (binary dummies)
        """
        # train the DNN on the adversarial examples for 10 epochs
        self.DNN.fit(
            advExmpls,
            classifications,
            batch_size=64,
            epochs=10,
            verbose=0
        )

    def testAccuracy(self, XTest, yTest, printPerf=True, returnCorrectIndex=False):
        """
        Method used to test the accuracy of the DNN on the given test data
        :param XTest:                   Test data
        :param yTest:                   Correct classification of the test data (binary dummies)
        :param printPerf:               Should accuracy be printed
        :param returnCorrectIndex:      Should list of booleans showing correct classification of image be returned
        :return:                        If returnCorrectIndex=False, number of correctly classified images
        """
        # put test data through DNN and get correct classifications
        pred = self.DNN.predict(XTest)
        correct = np.equal(np.argmax(pred,axis=1), np.argmax(yTest,axis=1))
        filter = np.logical_and(correct, np.any(pred>0.00001, axis=1))

        if printPerf:
            # print the accuracy of the DNN on the test data
            print(XTest.shape[0], np.sum(filter), 100*(np.sum(filter)/XTest.shape[0]))
        if returnCorrectIndex:
            # return the list showing which images are classified correctly
            return filter
        # return the number of correctly images
        return np.sum(filter)

    def testAccPerturb(self, XTest, yTest, perturbations, printPerf=True):
        """
        Method used to test the accuracy of the DNN when applying perturbation 2
        :param XTest:                   Test data
        :param yTest:                   Correct classification of the test data (binary dummies)
        :param perturbations:           List of perturbations to be used per digit
        :param printPerf:               Should accuracy be printed
        :return:
        """
        # put test data through DNN and get classifications
        pred = self.DNN.predict(XTest)
        # initialise list for perturbed images
        XPerturbed = [np.zeros(784)]
        for i in range(XTest.shape[0]):
            # get perturbation that corresponds to the predicted classification
            perturbation = perturbations[np.argmax(pred[i])]
            # apply the perturbation
            imagePerturbed = np.multiply(XTest[i], perturbation[0])
            imagePerturbed = np.add(imagePerturbed, perturbation[1])
            # add the perturbed image to the list
            XPerturbed = np.vstack((XPerturbed, imagePerturbed))
        # remove the row of zeros used for initialization
        XPerturbed = XPerturbed[1:]
        # classify the perturbed test data and get the correct classification
        pred = self.DNN.predict(XPerturbed)
        correct = np.equal(np.argmax(pred,axis=1), np.argmax(yTest,axis=1))
        filter = np.logical_and(correct, np.any(pred>0.00001, axis=1))

        if printPerf:
            # print the accuracy of the DNN
            print(XTest.shape[0], np.sum(filter), 100*(np.sum(filter)/XTest.shape[0]))
        return np.sum(filter)

    def save(self, filename):
        """
        Method used to save the DNN
        :param filename:    Name of the directory the DNN should be saved to
        """
        self.DNN.save("models\\" + filename)

    def writeWeights(self, filename):
        """
        Method used to write the weights and biases of the DNN to csv files
        :param filename:    Name of the file the weights should be written to
        """
        # delete the old weights file if there is one
        oldWeights = os.path.join(sys.path[0], filename)
        if os.path.exists(oldWeights):
            os.remove(oldWeights)

        # get the weights
        weights = self.DNN.weights

        # write them to the file for every layer
        for w in weights:
            with open(filename, "a") as f:
                # print the shape of the weights matrix/bias vector
                header = str(w.shape[0]) + ","
                if w.shape.__len__() == 2:
                    header += str(w.shape[1])
                else:
                    header += "1"
                f.write(header + "\n")
                np.savetxt(f, w, fmt='%s', delimiter=",")

    def writeTestdataPerformance(self, directory):
        """
        Method used to write 100 correctly classified instances from the test data to a csv file
        so they can be used for the performance tests of the bound tightening method
        """
        # get the correct classification of the test data
        y = np.argmax(self.yTest,axis=1)

        # predict the classification of the test data and get the correctly
        # classified instances that are actually classified (activation > 0)
        pred = self.DNN.predict(self.XTest)
        correct = np.equal(np.argmax(pred,axis=1), y)
        filter = np.logical_and(correct, np.any(pred!=0, axis=1))

        # get the first 100 correctly classified images and their classification
        images = self.XTest[filter][:100]
        classifications = y[filter][:100]

        # create a list of the files to be created
        files = [("\\images.csv", images), ("\\classifications.csv", classifications)]

        for file in files:
            # get the full directory of the file
            filename = dir + file[0]

            # delete the old files if necessary
            oldFile = os.path.join(sys.path[0], filename)
            if os.path.exists(oldFile):
                os.remove(oldFile)

            # write the data to the file
            with open(filename, "a") as f:
                f.write(str(file[1].shape[0]) + "\n")
                np.savetxt(f, file[1], fmt='%s', delimiter=",")

    def writeData(self):
        """
        Method used to create a training and test set of correctly classified images
        that is ordered by their classification
        """
        #  initialize the lists that will store the data
        correctImg = [np.zeros(784)]
        correctClass = []
        correctImgTest = [np.zeros(784)]
        correctClassTest = []
        incorrectImgTest = [np.zeros(784)]
        incorrectClassTest = []

        # get the correct classification of the test data
        y = np.argmax(self.yTest,axis=1)

        # for every digit:
        for i in range(0,10):
            # predict the images of the digit and get the correctly classified ones
            pred = self.DNN.predict(self.XTest[y == i])
            correct = np.argmax(pred,axis=1) == i
            filter = np.logical_and(correct, np.any(pred!=0, axis=1))

            # add 5 correctly classified images of the digit to the training data
            correctImg = np.vstack((correctImg, self.XTest[y == i][filter][0:5]))
            correctClass = np.append(correctClass, np.multiply(np.ones(5, dtype=int), i))

            # add 90 correctly classified images of the digit to the test data
            correctImgTest = np.vstack((correctImgTest, self.XTest[y == i][filter][5:95]))
            correctClassTest = np.append(correctClassTest, np.multiply(np.ones(90, dtype=int), i))

            # add 20 incorrectly classified images of the digit to the test data
            incorrectImgTest = np.vstack((incorrectImgTest, XTest[y == i][np.invert(filter)][0:20]))
            incorrectClassTest = np.append(incorrectClassTest, np.multiply(np.ones(20, dtype=int), i))

        # remove the row of zeros used to initialise the lists
        correctImg = correctImg[1:]
        correctImgTest = correctImgTest[1:]
        incorrectImgTest = incorrectImgTest[1:]
        # make the classification an integer
        correctClass = np.array(correctClass, dtype=int)
        correctClassTest = np.array(correctClassTest, dtype=int)
        incorrectClassTest = np.array(incorrectClassTest, dtype=int)

        # create a list of datasets that have to be written to files
        datasets = [
            ("\\train\\original", (correctImg, correctClass)),
            ("\\test\\original", (correctImgTest, correctClassTest)),
            ("\\test\\incorrectOriginal", (incorrectImgTest, incorrectClassTest))
        ]

        # for every dataset
        for dataset in datasets:
            # get the full directory of the dataset
            directory = "data" + dataset[0]
            # create a list of files that have to be created for the dataset
            files = [("\\images.csv", dataset[1][0]), ("\\classifications.csv", dataset[1][1])]
            for file in files:
                filename = directory + file[0]

                # delete the old file if necessary
                oldFile = os.path.join(sys.path[0], filename)
                if os.path.exists(oldFile):
                    os.remove(oldFile)

                # write the data
                with open(filename, "a") as f:
                    f.write(str(file[1].shape[0]) + "\n")
                    np.savetxt(f, file[1], fmt='%s', delimiter=",")

