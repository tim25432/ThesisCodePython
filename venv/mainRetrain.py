from DNNPretrained import DNNPretrained
import numpy as np


def main():
    """
    Main method used to test the accuracy of the retraining methods
    """
    # initialize the DNNs
    dnnBase = DNNPretrained("modelBasic")
    dnnOnlyAdv = DNNPretrained("modelOnlyAdv")
    dnnPROnlyAdv = DNNPretrained("modelBasic")
    dnnPRAdvOrig = DNNPretrained("modelBasic")
    dnnCR = DNNPretrained("modelCompRetrain")

    # get the training data
    advExmpls, advClass , origImg, origClass = getDatasets(False)

    advOrigImg = np.vstack((advExmpls,origImg))
    advOrigClass = np.vstack((advClass, origClass))

    # perform the partial retraining
    dnnPROnlyAdv.trainAdvExmpls(advExmpls, advClass)
    dnnPRAdvOrig.trainAdvExmpls(advOrigImg, advOrigClass)

    # get the test data
    advExmplsTest, advClassTest, origImgTest, origClassTest, origImgTestIncorrect, origClassTestIncorrect = getDatasets(True)

    # create a list of the models that have to be tested
    models = [
        ("base model", dnnBase),
        ("only adversarial(original weights/biases not used)", dnnOnlyAdv),
        ("partial retrain only adversarial", dnnPROnlyAdv),
        ("partial retrain adversarial+original", dnnPRAdvOrig),
        ("complete retrain", dnnCR)
    ]

    for model in models:
        # print the model that is currently being tested
        print(model[0])
        # test the accuracy on the different data sets
        adv = model[1].testAccuracy(advExmplsTest, advClassTest)
        orig1 = model[1].testAccuracy(origImgTest, origClassTest)
        orig2 = model[1].testAccuracy(origImgTestIncorrect, origClassTestIncorrect)
        # print the overall accuracy
        print ("overall: " + str(100*((adv + orig1 + orig2)/ 2000)))

def getDatasets(testSet):
    """
    Method used to get data sets of adversarial examples and original images
    :param testSet:     If the test data should be returned(if False training data is returned)
    :return:            Tuple of data sets
                        (Adversarial examples, binary correct classification of adversarial examples,
                        original correctly classified images, binary classification of images)
                        (if testSet also: original incorrectly classified images, binary classification of images))
    """
    # get the directory of the data sets
    directory = "data"
    if testSet:
        directory += "\\test"
    else:
        directory += "\\training"

    # get the adversarial examples and original correctly classified images
    advExmpls, advClass = getData(directory + "\\adversarial", False)
    origImg, origClass = getData(directory + "\\original", True)

    if testSet:
        # get the original incorrectly classified images
        origImgIncorrect, origClassIncorrect = getData(directory + "\\incorrectOriginal", True)
        return advExmpls, advClass, origImg, origClass, origImgIncorrect, origClassIncorrect

    return advExmpls, advClass, origImg, origClass

def getData(dir, skipRow):
    """
    Method used to get images and their correct classifications of a data set
    :param dir:         Directory the data set can be found in
    :param skipRow:     If the first row of the data should be skipped
    :return:            Tuple of two lists (images, binary classification)
    """
    skip = 0
    if skipRow:
        skip = 1

    # load the images and classifications
    images = np.loadtxt(open(dir + "\\images.csv", "rb"), delimiter=",", skiprows=skip)
    classifications = np.loadtxt(open(dir + "\\classifications.csv", "rb"), delimiter=",", skiprows=skip)
    # convert the classifications into binary dummies
    classBin = getBinaryClass(classifications)
    return images, classBin

def getBinaryClass(classifications):
    """
    Method used to convert classifications into binary dummies
    :param classifications:     List of classifications
    :return:                    List of binary dummies of classifications
    """
    # create list of zeros for every classification
    classificationsBin = np.zeros((classifications.shape[0],10))
    for i in range(0,classifications.shape[0]):
        # set binary dummy corresponding to the correct classification to 1
        classificationsBin[i][int(classifications[i])] = 1
    return classificationsBin


if __name__ == "__main__":
    main()
