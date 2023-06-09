# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
# Define data (Given)
X=np.array([[-1.51], [-1.29], [-1.18], [-0.64],
[-0.53], [-0.09], [0.13], [0.35],
[0.89], [1.11], [1.33], [1.44]]) 

y=np.array([[0], [0], [0], [0],
[1], [1], [1], [1],[0], [0], [0], [0]])

# Display data
plt.scatter(X,y)

# Define sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Define sigmoid prime function (derivative of sigmoid)
def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

# Pass input data through sigmoid function
# Add a weight to the sigmoid function
# Alter and tune the weight to separate the left portion of data
w11 = 20 # -20
w12 = 5 # -8
# Add weights for the output layer
w21 = 15 # -20
w22 = 10 # 17
# Add a bias term
b11 = 1 # -12
b12 = 1 # 4.5
b22 = -8 # -8
# Define matrices for the weight and bias parameters
W1 = np.array([[w11, w12]]) 
W2 = np.array([[w21], [w22]])
B1 = np.array([[b11, b12]])
B2 = np.array([[b22]])

# Choose a parameter. Let's say w11. Define the range in which to vary the parameter
N = np.linspace(-100,100,10000)
Cost = []

# ax = plt.axes(projection='3d')

def cost(y, out):
    return -np.sum((np.square(y) - np.square(out)))/len(y)

# Express outputs in matrices
def feedforward(x, w1, w2, b1, b2):
    a1 = sigmoid(np.dot(x, w1) + b1) # X = 12X1, W1 = 1X2, B1 = 1X2
    out = sigmoid(np.dot(a1, w2) + b2) # A1 = 12X2, W2 = 2X1
    return a1, out

# A1, Out = feedforward(X, W1, W2, B1, B2)

def backprop(y, out, w1, w2, b1, b2, a1, X):
    # Backpropagation 
    # Taking derivative w.r.t. parameter w11
    # Derivates for 7 parameters 
    # dL_dw11, dL_dw12, dL_dw21, dL_dw22, dL_dw_b11, dL_dw_b12, dL_dw_b22

    # dL_dW1 = (dL_dy)*(dy_dz2)*(dz2_da1)*(da1_dz1)*(dz1_dW1)
    # dL_dW2 = (dL_dy)*(dy_dz2)*(dz2_da1)*(dz2_dW2)
    # dL_dB1 = (dL_dy)*

    # dL_dy = -((out/y)-(1-out)/(i-y))
    # dy_dz2 = y(1 - y)
    # Substituting these values in the main expression
    # dL_dw1 = (y - out)*W2*a1*(1 - a1)
    # dL_db1 = (y - out)*W2*a1*(1 - a1)

    lr = 0.1

    res=out-y
    d_w2 = np.dot(a1.T, res) # dL_dW2
    d_b2 = np.sum(res) # dL_dB2
    d_b1_v=np.dot(res, w2.T) * a1*(1-a1)
    d_b1 = np.sum(d_b1_v,axis=0)
    d_w1 = np.dot(X.T, d_b1_v)

    # Output
    w1 = w1 - d_w1*lr
    w2 = w2 - d_w2*lr
    b1 = b1 - d_b1*lr
    b2 = b2 - d_b2*lr

    return w1, w2, b1, b2
cost_ = 0
for i in range(len(N)):
    # W1 = np.array([[N[i], w12]]) # varying w11
    a1, out = feedforward(X, W1, W2, B1, B2)
    cost_ = cost(y, out)
    Cost.append(cost_)
    W1, W2, B1, B2 = backprop(y, out, W1, W2, B1, B2, a1, X)
    if (cost_ > -0.001 and cost_ < 0.001):
        print("Converged!")
        exit()
    plt.scatter(X,y)
    plt.plot(X,a1[:,0], c = 'red')
    plt.plot(X,a1[:,1], c = 'blue')
    plt.plot(X,out, c = 'black')
    plt.text(-0.5,1,"Step =" + str(i))
    plt.text(0.5,1,"Cost =" + str(cost_))
    plt.text(-1.5,1,"W1 =" + str(W1))
    plt.text(-1.5,0.9,"W2 =" + str(W2))
    plt.text(-1.5,0.8,"B1 =" + str(B1))
    plt.text(-1.5,0.7, "B2 =" + str(B2))
    # plt.draw()
    plt.pause(0.05)
    plt.clf()
# plt.show()

# Plot the output of the sigmoid function
# plt.plot(X,A1[:,0], c = 'red')
# plt.plot(X,A1[:,1], c = 'blue')
# plt.plot(X,Out, c = 'black')
# plt.show()


# Plot the cost function for the parameter w11
plt.plot(N, Cost, c = 'black')
plt.show()

# Plot the cost function for the parameters w11 and w12
# ax.plot3D(N, Cost_W11, Cost_W12, 'gray')
# plt.show()
