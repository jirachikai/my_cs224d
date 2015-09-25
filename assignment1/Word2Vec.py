# Implement your skip-gram and CBOW models here
import numpy as np
import random
from Softmax import softmax
from NeuralNetworkBasics import sigmoid,sigmoid_grad,gradcheck_naive
from cs224d.data_utils import *

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component) (V_wi)                         #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       #
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    V, D = outputVectors.shape

    scores = softmax(outputVectors.dot(predicted).reshape(1,V)).reshape(V,)
    cost = -np.log(scores[target])

    labels = np.zeros(V)
    labels[target] = 1
    dscores = scores - labels
    gradPred = dscores.dot(outputVectors)
    grad = dscores.reshape(V, 1).dot(predicted.reshape(D, 1).T)
    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    sampleIndexs = []
    while True:
        index = dataset.sampleTokenIdx()
        if index != target:
            sampleIndexs.append(index)
        if len(sampleIndexs)>=K:
            break
#    for i in xrange(K):
#        index = dataset.sampleTokenIdx()
#        sampleIndexs.append(index)

    sampleVectors = outputVectors[sampleIndexs, :]
    w_r_out = sigmoid(outputVectors[target].dot(predicted))
    w_r_k = sigmoid(- sampleVectors.dot(predicted))
    cost = -np.log(w_r_out)-np.sum(np.log(w_r_k))

    gradPred = outputVectors[target] * ( w_r_out - 1 ) + (1 - w_r_k).dot(sampleVectors)
    grad = np.zeros(outputVectors.shape)
    grad[target] = predicted *(w_r_out - 1)
    for i in xrange(K):
        grad[sampleIndexs[i]] += predicted * (1 - w_r_k)[i]
    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    currentWordIndex = tokens[currentWord]
    predicted = inputVectors[currentWordIndex]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        _cost, _gradPred, _grad = word2vecCostAndGradient(predicted,tokens[word],outputVectors)
        cost += _cost
        gradIn[currentWordIndex] += _gradPred
        gradOut += _grad
    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    currentWordIndex = tokens[currentWord]
    predicted = outputVectors[currentWordIndex]
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        _cost, _gradPred, _grad = word2vecCostAndGradient(predicted,tokens[word],inputVectors)
        cost += _cost
        gradOut[currentWordIndex] += _gradPred
        gradIn += _grad
    ### END YOUR CODE

    return cost, gradIn, gradOut

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """
    ### YOUR CODE HERE
    N = x.shape[0]
    x /= np.sqrt(np.sum(x ** 2, axis=1)).reshape(N, 1)    
    ### END YOUR CODE

    return x

# Test this function
#print "=== For autograder ==="
#print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]

# Gradient check!

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

#random.seed(31415)
#np.random.seed(9265)
#dummy_vectors = normalizeRows(np.random.randn(10,3))
#dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
#print "==== Gradient check for skip-gram ===="
#gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
#gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
#print "\n==== Gradient check for CBOW      ===="
#gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
#gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
#
#print "\n=== For autograder ==="
#print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
#print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
#print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
#print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)

# Now, implement SGD

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

import glob
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    # find recently file
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    #   - PRINT_EVERY: specifies every how many iterations to output  #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        ### YOUR CODE HERE
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
        cost, grad = f(x)
        x = x - step * grad
        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            print "iter " + str(iter) + ". Cost = " + str(cost)
        ### END YOUR CODE

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x

# Load some data and initialize word vectors

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Train word vectors (this could take a while!)

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
#init the parameter, random.rand(nWords,dimenVectors): generating a nWords*dimenVectors matrix.
#generate two nWords*dimenVectors matrix, one for input, one for output.
#wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors,
#                              np.zeros((nWords, dimVectors))), axis=0)
#wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient),
#                   wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
# sanity check: cost at convergence should be around or below 10

# sum the input and output word vectors
#wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

#print "\n=== For autograder ==="
#checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
#checkIdx = [tokens[word] for word in checkWords]
#checkVecs = wordVectors[checkIdx, :]
#print checkVecs

# Visualize the word vectors you trained
#_, wordVectors0, _ = load_saved_params()
#wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

#import matplotlib.pyplot as plt
#visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"]
#visualizeIdx = [tokens[word] for word in visualizeWords]
#visualizeVecs = wordVectors[visualizeIdx, :]
#temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
#covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
#U,S,V = np.linalg.svd(covariance)
#coord = temp.dot(U[:,0:2])
#
#for i in xrange(len(visualizeWords)):
#    plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
#
#plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
#plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))
#plt.show()