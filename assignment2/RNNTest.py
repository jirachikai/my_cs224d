import sys, os
from numpy import *
#from matplotlib.pyplot import *

from rnnlm import RNNLM
# Gradient check on toy data, for speed
#random.seed(10)
#wv_dummy = random.randn(10,50)
#model = RNNLM(L0 = wv_dummy, U0 = wv_dummy,
#              alpha=0.005, rseed=10, bptt=4)
#model.grad_check(array([1,2,3]), array([2,3,4]))

from data_utils import utils as du
import pandas as pd

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)
##
# Below needed for 'adj_loss': DO NOT CHANGE
fraction_lost = float(sum([vocab['count'][word] for word in vocab.index
                           if (not word in word_to_num)
                               and (not word == "UUUNKKK")]))
fraction_lost /= sum([vocab['count'][word] for word in vocab.index
                      if (not word == "UUUNKKK")])
print "Retained %d words from %d (%.02f%% of all tokens)" % (vocabsize, len(vocab),
                                                             100*(1-fraction_lost))

# Load the training set
docs = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs, word_to_num)
X_train, Y_train = du.seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs, word_to_num)
X_dev, Y_dev = du.seqs_to_lmXY(S_dev)

# Load the test set (final evaluation only)
docs = du.load_dataset('data/lm/ptb-test.txt')
S_test = du.docs_to_indices(docs, word_to_num)
X_test, Y_test = du.seqs_to_lmXY(S_test)

# Display some sample data
#print " ".join(d[0] for d in docs[7])
#print S_test[7]
#
#hdim = 100 # dimension of hidden layer = dimension of word vectors
#random.seed(10)
#L0 = zeros((vocabsize, hdim)) # replace with random init,
                              # or do in RNNLM.__init__()
# test parameters; you probably want to change these
#model = RNNLM(L0, U0 = L0, alpha=0.1, rseed=10, bptt=1)
# Gradient check is going to take a *long* time here
# since it's quadratic-time in the number of parameters.
# run at your own risk... (but do check this!)
# model.grad_check(array([1,2,3]), array([2,3,4]))

#### YOUR CODE HERE ####

##
# Pare down to a smaller dataset, for speed
# (optional - recommended to not do this for your final model)
hdim = 100 # dimension of hidden layer = dimension of word vectors
random.seed(10)
L0 = zeros((vocabsize, hdim)) # replace with random init,
L0 = 0.1 * random.randn(*L0.shape)   # or do in RNNLM.__init__()
# test parameters; you probably want to change these
model = RNNLM(L0, U0 = L0, alpha=0.1, rseed=10, bptt=3)
ntrain = len(Y_train)
X = X_train[:ntrain]
Y = Y_train[:ntrain]
k = 5
indices = range(ntrain)
def idxiter_batches():
    num_batches = ntrain / k
    for i in xrange(num_batches):
        yield random.choice(indices, k)

model_output = model.train_sgd(X=X, y=Y, idxiter=idxiter_batches(), printevery=100, costevery=10000)

dev_loss = model.compute_mean_loss(X_dev, Y_dev)
## DO NOT CHANGE THIS CELL ##
# Report your numbers, after computing dev_loss above.
def adjust_loss(loss, funk, q, mode='basic'):
    if mode == 'basic':
        # remove freebies only: score if had no UUUNKKK
        return (loss + funk*log(funk))/(1 - funk)
    else:
        # remove freebies, replace with best prediction on remaining
        return loss + funk*log(funk) - funk*log(q)
# q = best unigram frequency from omitted vocab
# this is the best expected loss out of that set
q = vocab.freq[vocabsize] / sum(vocab.freq[vocabsize:])
print "Unadjusted: %.03f" % exp(dev_loss)
print "Adjusted for missing vocab: %.03f" % exp(adjust_loss(dev_loss, fraction_lost, q))

# Save to .npy files; should only be a few MB total
assert(min(model.sparams.L.shape) <= 100) # don't be too big
assert(max(model.sparams.L.shape) <= 5000) # don't be too big
save("rnnlm.L.npy", model.sparams.L)
save("rnnlm.U.npy", model.params.U)
save("rnnlm.H.npy", model.params.H)
#dev_loss = model.compute_mean_loss(X_dev, Y_dev)
#### END YOUR CODE ####

def seq_to_words(seq):
    return [num_to_word[s] for s in seq]

seq, J = model.generate_sequence(word_to_num["<s>"],
                                 word_to_num["</s>"],
                                 maxlen=100)
print J
# print seq
print " ".join(seq_to_words(seq))

# Replace UUUNKKK with a random unigram,
# drawn from vocab that we skipped
from nn.math import MultinomialSampler, multinomial_sample

def fill_unknowns(words):
    #### YOUR CODE HERE ####
    ret = words
    for i in xrange(len(ret)):
        if ret[i] == 'UUUNKKK':
            index = multinomial_sample(vocab.freq)
            ret[i] = list(vocab.index)[index]

    #### END YOUR CODE ####
    return ret

print " ".join(fill_unknowns(seq_to_words(seq)))