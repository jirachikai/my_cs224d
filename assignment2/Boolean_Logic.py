from numpy import *
from matplotlib.pyplot import *

matplotlib.rcParams['savefig.dpi'] = 100
colors = 'rbcm'
markers = 'soos'
npts = 4 * 40; random.seed(10)
x = random.randn(npts)*0.1 + array([i & 1 for i in range(npts)])
y = random.randn(npts)*0.1 + array([(i & 2) >> 1 for i in range(npts)])
data = vstack([x,y])

def show_pts(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[0,idx], data[1,idx],
             marker=markers[i], linestyle='.',
             color=colors[i], alpha=0.5)
    gca().set_aspect('equal')

def show_pts_1d(data):
    for i in range(4):
        idx = (arange(npts) % 4 == i)
        plot(data[idx], marker=markers[i], linestyle='.',
             color=colors[i], alpha=0.5)
    gca().set_aspect(npts/4.0)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

figure(figsize=(4,4)); show_pts(data); ylim(-0.5, 1.5); xlim(-0.5, 1.5)
xlabel("x"); ylabel("y"); title("Input Data")
show()
x = linspace(-1, 1, 100); figure(figsize=(4,3))
plot(x, sigmoid(x), 'k', label="$\sigma(x)$");
plot(x, sigmoid(5*x), 'b', label="$\sigma(5x)$");
plot(x, sigmoid(15*x), 'g', label="$\sigma(15x)$");
legend(loc='upper left'); xlabel('x');
show()

W = zeros((2,2))
b1 = zeros((2,1))
U = zeros(2)
b2 = 0

#### YOUR CODE HERE ####
z = 5 # control gate steepness
W = array([[-1, -1],[1, 1]])
b1 = array([[1.5],[-0.5]])
U = array([1, 1])
b2 = -1.5
#### END YOUR CODE ####

# Feed-forward computation
h = sigmoid(z*(W.dot(data) + b1))
p = sigmoid(z*(U.dot(h) + b2))

# Plot hidden layer
subplot(1,2,1); show_pts(h)
title("Hidden Layer"); xlabel("$h_1$"); ylabel("$h_2$")
ylim(-0.1, 1.1); xlim(-0.1, 1.1)
# Plot predictions
subplot(1,2,2); show_pts_1d(p)
title("Output"); ylabel("Prediction"); xticks([])
axhline(0.5, linestyle='--', color='k')
tight_layout()
show()