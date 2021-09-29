from DNNPretrained import DNNPretrained
from mainRetrain import getDatasets
import numpy as np


def main():
    """
    Main method used to test the accuracy after applying the perturbation approaches
    """
    # load the DNN
    dnn = DNNPretrained("modelBasic")

    # get the perturbations
    p1, q1 = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbation.csv", delimiter=","))
    p1p = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbationWeights.csv", delimiter=","))
    q1q = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbationDisturbances.csv", delimiter=","))
    p1MinDist, q1MinDist = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbationMinDist.csv", delimiter=","))

    perturbation2 = getPerturbations()

    # get the test data
    advExmpls, advClass , origImg, origClass, origImgIncorrect, origClassIncorrect = getDatasets(True)

    # create a list of the variations of perturbation 1 and their names
    perturbations = [
        (None, None, "regular"),
        (p1, q1, "perturbation 1"),
        (p1p, None, "only weights"),
        (None, q1q, "only disturbances"),
        (p1MinDist, q1MinDist, "min distance")
    ]

    for perturbation in perturbations:
        # perturb the image
        advExmplsPerturbed = advExmpls
        origImgPerturbed = origImg
        origImgIncorrectPerturbed = origImgIncorrect

        if perturbation[0] is not None:
            advExmplsPerturbed = np.multiply(advExmplsPerturbed, perturbation[0])
            origImgPerturbed = np.multiply(origImgPerturbed, perturbation[0])
            origImgIncorrectPerturbed = np.multiply(origImgIncorrectPerturbed, perturbation[0])

        if perturbation[1] is not None:
            advExmplsPerturbed = np.add(advExmplsPerturbed, perturbation[1])
            origImgPerturbed = np.add(origImgPerturbed, perturbation[1])
            origImgIncorrectPerturbed = np.add(origImgIncorrectPerturbed, perturbation[1])

        # test the accuracy
        print(perturbation[2])
        adv = dnn.testAccuracy(advExmplsPerturbed, advClass)
        orig1 = dnn.testAccuracy(origImgPerturbed, origClass)
        orig2 = dnn.testAccuracy(origImgIncorrectPerturbed, origClassIncorrect)
        print ("overall: " + str(100*((adv + orig1 + orig2)/ 2000)))

    # test the accuracy when applying perturbation 2
    print("perturbation 2")
    adv = dnn.testAccPerturb(advExmpls, advClass, perturbation2)
    orig1 = dnn.testAccPerturb(origImg, origClass, perturbation2)
    orig2 = dnn.testAccPerturb(origImgIncorrect, origClassIncorrect, perturbation2)
    print ("overall: " + str(100*((adv + orig1 + orig2)/ 2000)))

def getPerturbations():
    """
    Method used to get the perturbations for every digit from perturbation 2
    :return:    List of 10 perturbations corresponding to each digit
    """
    perturbations = []
    for i in range(10):
        # read the perturbation for every digit
        perturbations += [np.transpose(np.loadtxt("input\\perturbations\\perturbation2\\perturbation" + str(i) + ".csv", delimiter=","))]
    return np.array(perturbations)

if __name__ == '__main__':
    main()
