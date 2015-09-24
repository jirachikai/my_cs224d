import numpy as np
def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################

    ### YOUR CODE HERE
    N = x.shape[0]
    x -= np.max(x, axis=1).reshape(N, 1)
    x = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(N, 1)
    return x
    ### END YOUR CODE

#print softmax(np.array([[1,2],[3,4]]))
#print "=== For autograder ==="
#print softmax(np.array([[1001,1002],[3,4]]))
#print softmax(np.array([[-1001,-1002]]))

