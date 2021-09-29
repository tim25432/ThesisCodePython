from DNNPretrained import DNNPretrained
from mainRetrain import getDatasets
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Main method used to create several images of adversarial examples and the perturbation
    """
    # create images of several adversarial examples
    folders = [
        "adversarial",
        "adversarialEps",
        "adversarialMaxDev",
        "adversarialMaxDevEps",
        "original"
    ]
    # create images for all digits and the feature visualization
    files = ["input\\examples\\%s\\csv\\%sto%s.csv" % (j, i, (i + 5)%10) for j in folders for i in range(0,10)]
    files += ["input\\examples\\visualization\\csv\\visualize%s.csv" % (i) for i in range(0,10)]
    for file in files:
        createImage(file)

    # get the perturbations
    perturbation1 = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbation.csv", delimiter=","))
    perturbationMinDist = np.transpose(np.loadtxt("input\\perturbations\\perturbation1\\perturbationMinDist.csv", delimiter=","))

    # create images for perturbation 1
    print("perturbation 1")
    createPerturbationImages(perturbation1, "output\\perturbationImage\\perturbation1")
    createPerturbedImage(perturbation1, "output\\perturbationImage\\perturbation1\\example")

    # create images for perturbation 1 with minimum distance
    print("perturbation min distance")
    createPerturbationImages(perturbationMinDist, "output\\perturbationImage\\minDist")
    createPerturbedImage(perturbationMinDist, "output\\perturbationImage\\minDist\\example")

def createImage(filename):
    """
    Method used to create a png image from a csv file
    containing a gray scaled image
    :param filename:    Name of the csv file containing the image
    """
    # get the image from the csv file
    x = np.loadtxt(open(filename, "rb"), delimiter=",")

    # create a figure
    fig = plt.figure()
    fig.set_size_inches(5, 5)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # put the image in the figure and save it as a png file
    ax.imshow(x, cmap="gray", aspect='auto')
    png = filename.replace("csv", "png")
    fig.savefig(png)

def createPerturbationImages(perturbation, outputDirectory):
    """
    Method used to create images portraying a perturbation's
    weights and disturbance and the perturbation itself
    :param perturbation:        The perturbation to be portrayed
    :param outputDirectory:     The directory the images should be saved to
    """
    # split up the weights and disturbances of the perturbation
    p, q = perturbation
    # resize them from a 1d-array to a 2d-array so they can be displayed as images
    p = np.resize(p, (28, 28))
    q = np.resize(q, (28, 28))

    # create a figure
    fig = plt.figure()
    fig.set_size_inches(5, 5)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # create an image of the weights
    print("weights interval: ", p.min(), p.max())
    ax.imshow(p, cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\weights.png")

    # create an image that shows what colour the unperturbed image would be
    ax.imshow([[p.min(), 1, p.max()]], cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\aScaleWeights.png")

    # do the same for the disturbances
    print("disturbances interval: ", q.min(), q.max())
    ax.imshow(q, cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\disturbances.png")

    ax.imshow([[q.min(), 0, q.max()]], cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\aScaleDisturbances.png")

    # do the same for the full perturbation
    q = np.add(p, q)
    print("disturbances interval: ", q.min(), q.max())
    ax.imshow(q, cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\perturbation.png")

    ax.imshow([[q.min(), 1, q.max()]], cmap="gray", aspect='auto')
    fig.savefig(outputDirectory + "\\aScalePerturbation.png")

def createPerturbedImage(perturbation, outputDirectory):
    """
    Method used to create perturbed images of adversarial examples
    as well as images of the adversarial examples and their original image
    :param perturbation:        The perturbation that should be used
    :param outputDirectory:     The directory the images should be saved to
    """
    # get the DNN
    dnn = DNNPretrained("modelBasic")

    # get the adversarial examples and perturb them
    advExmpls, advClass , origImg, _, _, _ = getDatasets(True)
    advExmplsPerturbed = np.multiply(advExmpls, perturbation[0])
    advExmplsPerturbed = np.add(advExmplsPerturbed, perturbation[1])

    # get which perturbed adversarial examples were correctly classified
    correct = dnn.testAccuracy(advExmplsPerturbed, advClass, returnCorrectIndex=True)
    # for every digit:
    for digit in range(10):
        # check if any of the adversarial examples of this digit are classified correctly
        i = digit*90
        if np.sum(correct[i:i+90]) == 0:
            continue

        print(digit)

        # get the index of the first correctly classified perturbed adversarial example
        firstCorrectIndex = i + np.where(correct[i:i+90] == True)[0][0]

        # get the first first correctly classified perturbed adversarial example,
        # the unperturbed adversarial example and the original image
        perturbedAdvExmpl = np.resize(advExmplsPerturbed[firstCorrectIndex], (28, 28))
        advExmpl = np.resize(advExmpls[firstCorrectIndex], (28, 28))
        origImgIndex = int(firstCorrectIndex/9)
        origImgIndex = 90 * int(origImgIndex/10) + origImgIndex % 10
        original = np.resize(origImg[origImgIndex], (28, 28))

        # make a list of the images that have to be created
        imgsToMake = [
            (perturbedAdvExmpl, "\\" + str(digit) + "\\perturbed.png"),
            (advExmpl, "\\" + str(digit) + "\\adversarial.png"),
            (original, "\\" + str(digit) + "\\original.png")
                      ]

        # create a figure
        fig = plt.figure()
        fig.set_size_inches(5, 5)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # for all images that have to be created:
        for image in imgsToMake:
            # print the images bounds and create the image and a legend
            print(image[1], image[0].min(), image[0].max())
            ax.imshow(image[0], cmap="gray", aspect='auto')
            fig.savefig(outputDirectory + image[1])

            ax.imshow([[image[0].min(), 1, image[0].max()]], cmap="gray", aspect='auto')
            fig.savefig(outputDirectory + image[1].replace(".png", "Scale.png"))


if __name__ == "__main__":
    main()
