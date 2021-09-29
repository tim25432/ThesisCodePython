import keras
import numpy as np
import os
import sys
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.random import set_seed
from random import seed


class DNN:
    """
    Class used to train the DNN and save the weights and biases
    """
    def __init__(self, XTrain, yTrain, XTest, yTest, XVal, yVal, architecture, learningRate):
        """
        Initializes the DNN and trains it on the training data
        :param XTrain:          Training data
        :param yTrain:          Correct classification of the training data (binary dummies)
        :param XTest:           Test data
        :param yTest:           Correct classification of the test data (binary dummies)
        :param XVal:            Data used for validation
        :param yVal:            Correct classification of the validation data (binary dummies)
        :param architecture:    Architecture of the DNN
        :param learningRate:    Learning rate used during the training process
        """
        # set the seeds to get consistent results
        set_seed(0)
        seed(0)

        # store the data
        self.XTrain, self.yTrain = XTrain, yTrain
        self.XTest, self.yTest = XTest, yTest
        self.XVal, self.yVal = XVal, yVal
        self.architecture = architecture

        # create the DNN
        self.DNN = Sequential()
        # add the input layer
        self.DNN.add(keras.Input(shape=(self.XTrain[0].size,)))
        # add the hidden layers
        for n_k in architecture:
            self.DNN.add(Dense(n_k, activation='relu'))

        # add the output layer
        self.DNN.add(Dense(10, activation='relu'))

        # set the settings for the training process
        self.DNN.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learningRate),
            # Loss function to minimize
            loss=keras.losses.MeanSquaredError(),
            # List of metrics to monitor
            metrics=[keras.metrics.CategoricalAccuracy()]
        )

        # train the DNN
        self.fitDNN()

    def fitDNN(self):
        """
        Method used to train the DNN on the training data
        """
        # train the DNN for 50 epochs
        history = self.DNN.fit(
            self.XTrain,
            self.yTrain,
            batch_size=64,
            epochs=50,
            validation_data=(self.XVal, self.yVal),
            verbose=0 # stop keras from printing the progress for each epoch
        )

        # print the test set accuracy of the DNN
        results = self.DNN.evaluate(self.XTest, self.yTest, batch_size=64, verbose=0)
        print(str(self.architecture) + ": " + str(history.history["categorical_accuracy"][-1]) + " " + str(results[1]))

    def writeWeights(self, filename):
        """
        Method used to write the weights and biases of the DNN to csv files
        :param filename:    Name of the file the weights should be written to
        """
        # get the directory of the weights file
        filepath = "output\\weights\\"
        for i in self.architecture:
            filepath += str(i) + "_"
        filepath = filepath[:-1] + "\\" + filename

        # delete the old weights file if there is one
        oldWeights = os.path.join(sys.path[0], filepath)
        if os.path.exists(oldWeights):
            os.remove(oldWeights)

        # get the weights
        weights = self.DNN.weights

        # write them to the file for every layer
        for w in weights:
            with open(filepath, "a") as f:
                # print the shape of the weights matrix/bias vector
                header = str(w.shape[0]) + ","
                if w.shape.__len__() == 2:
                    header += str(w.shape[1])
                else:
                    header += "1"
                f.write(header + "\n")
                np.savetxt(f, w, fmt='%s', delimiter=",")

    def writeTestdataPerformance(self):
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

        # set the directory of the file
        archString = ""
        for i in self.architecture:
            archString += str(i) + "_"
        archString = archString[:-1]

        dir = "output\\testdataPerformance\\" + archString

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

    def save(self, filename):
        """
        Method used to save the DNN
        :param filename:    Name of the directory the DNN should be saved to
        """
        self.DNN.save("models\\" + filename)
