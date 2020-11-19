#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    x_split = x.split()
    features = collections.defaultdict(int)
    for y in x_split:
        features[y] += 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, validationExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    m = len(trainExamples)
    # Forward prop
    for i in range(numIters):
        for k in range(m):
            x, y = trainExamples[k]
            phi = featureExtractor(x)
            margin = dotProduct(weights, phi) * y
            if margin < 1:
                increment(weights, eta*y, phi)

        train = evaluatePredictor(trainExamples, lambda example_x: 1 if dotProduct(weights, featureExtractor(example_x)) > 0 else -1)
        validate = evaluatePredictor(validationExamples, lambda example_x: 1 if dotProduct(weights, featureExtractor(example_x)) > 0 else -1)
        print('iter', i, 'train', train, 'validation', validate)

    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.

    # Note that the weight vector can be arbitrary during testing. 
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = dict.fromkeys(weights.keys(), random.random())
        y = 1 if dotProduct(weights, phi) > 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        runon = "".join(x.split())
        grams = {}
        for i in range(len(x)-n):
            if runon[i:i+n] not in grams:
                grams[runon[i:i+n]] = 1
            else:
                grams[runon[i:i + n]] += 1
        return grams
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    # Store all possible elements for x and y values in two lists
    # so that we can randomly pull from them later to generate centroids
    # possible_ys = []
    # possible_xs = []
    # for example in examples:
    #     possible_ys += example.values()
    #     possible_xs += example.keys()
    # # One possible source of error is that centroids can be duplicate
    # # values, even though sample is taken without replacement
    # centroids = random.sample(possible_xs, K), random.sample(possible_ys, K)
    # centroids = dict(enumerate())

    # while not len(centroids) == K:
    #     centroids.items()

    cost = 0
    loss = 0
    # Randomly assign centroids
    centroids = {i: random.choice(examples).copy() for i in range(K)}
    for y in range(maxIters):
        # Finish early if no more movement possible
        if cost == loss:
            break

        cost = loss
        loss = 0

    def whichCentroidAreYou():
        output = {}
        exampleCoordinates = [3, 7, 9]
        centroidCoordinates = [5, 1, 11]

        def difference(example, centroid):
            return list(x - y for x, y in zip(example, centroid))

        def addSquares(list):
            return sum(i ** 2 for i in list)

        for z in range(len(examples)):
            # Centroid coordinates

            # Each example should have a list of distances to each centroid
            normalDistances = []
            # For each centroid, compute the distance and store
            for k in range(K):
                normalDistances += addSquares(difference(exampleCoordinates, centroidCoordinates))
            # Takes the lengths to centroids and puts them in a numbered dict
            dictLengthsToCentroids = dict(enumerate(normalDistances, start=1))
            # returns the key (centroid) with the smallest value (distance)
            label = min(dictLengthsToCentroids, key=dictLengthsToCentroids.get)
            # Assigns each example to the closest centroid
            output['z' + str(z+1)] = label
        return output


    def centroidMigration(x):
        result = {}


        return result

    for x in range(maxIters):
        ##### establish random centroids ########
        centroid_assmt = whichCentroidAreYou(examplesDict, centroidsDict)

    res = centroidMigration(dict((v, k) for k, v in centroid_assmt.items()))

    # END_YOUR_CODE





















