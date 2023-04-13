import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from MLP_visualize import mlpVisualize

num_inputs = 2
hidden_layers = [3,2]
num_outputs = 1

epochs = 1000
mu = 10

layers = [num_inputs] + hidden_layers + [num_outputs]
img = mlpVisualize(layers)

weights = []
for i in range(len(layers) - 1):
    w = np.random.rand(layers[i], layers[i + 1])/2
    weights.append(w)

w_derivatives = []
for i in range(len(layers) - 1):
    d = np.zeros((layers[i], layers[i + 1]))
    w_derivatives.append(d)

activations = []
for i in range(len(layers)):
    a = np.zeros(layers[i])
    activations.append(a)


def forward_propagate(input):
    a = input
    activations[0] = a #input layer
    for i, w in enumerate(weights):
        h = np.dot(a, w) # hidden layers -
        a = sigmoid(h) # (perceptron -> sigmoid)
        activations[i + 1] = a
    return h # output layer -> LNU


def back_propagate(error):

    # dE/dW[i] = ((y - a[i+1]) * s'(h[i+1])) * a[i]
    #      ->   s'(h[i+1]) = s(h[i+1]) * (1-s(h[i+1]))
    #      ->   s(h[i+1]) = a[i+1]
    # dE/dW[i-1] = ((y - a[i+1]) * s'(h[i+1])) * W[i] * s'(h[i]) *  a[i-1]

    for i in reversed(range(len(w_derivatives))):
        a = activations[i + 1]
        dsig = a * (1.0 - a)
        delta = error * dsig
        delta_reshaped = delta.reshape(delta.shape[0], -1).T
        current_a = activations[i].reshape(activations[i].shape[0], -1)
        w_derivatives[i] = np.dot(current_a, delta_reshaped)
        error = np.dot(delta, weights[i].T)
    return


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradient_descent(mu):
    for i in range(len(weights)):
        weights[i] += w_derivatives[i] * mu
    return


def train(inputs, targets, epochs, mu):
    kk = 0
    for i in range(epochs):
        sum_err = 0
        for j, input in enumerate(inputs):
            target = targets[j]
            output = forward_propagate(input)
            err = target - output 
            e_all[kk] = err
            w_all.append(copy.deepcopy(weights))
            back_propagate(err)
            gradient_descent(mu)
            sum_err += np.average((target - output)**2)
            kk += 1
        SSE[i] = sum_err

        print('Error: {} at epoch {}'.format(sum_err / len(inputs), i))
    return


hm = x1 = np.array([5.5,  7.5,  9.5,  12,  15,  18.5, 23, 29.5, 36])
vek = x2 = np.array([4.5, 6, 9, 1.5*12, 2.5*12, 4.5*12,  9*12,  13.5*12,   15*12])  # vek v mesicich
davka = y = np.array([3,    4,    5,    6,   8,   10,   13, 16,   20])
N = len(x1)

# min max norm
x1_min = min(x1)
x2_min = min(x2)
y_min = min(y)
x1_max = max(x1)
x2_max = max(x2)
y_max = max(y)
x1 = (x1 - x1_min) / (x1_max - x1_min)
x2 = (x2 - x2_min) / (x2_max - x2_min)
y = (y - y_min) / (y_max - y_min)

inputs = np.array([x1, x2]).T
targets = y

w_all = []
e_all = np.zeros((epochs * N)) 
SSE = np.zeros(epochs)

#train
train(inputs, targets, epochs, mu)

#test
yn = np.zeros(N)
for i in range(N):
    input = [hm[i], vek[i]]
    # minmaxnorm
    x1 = (input[0] - x1_min)/(x1_max - x1_min)
    x2 = (input[1] - x2_min)/(x2_max - x2_min)
    output = forward_propagate([x1, x2])
    yn[i] = output
    # un-z-score
    output = (output * (y_max - y_min)) + y_min
    print('pro hm = {} kg a vek = {} mesicu je predikovana davka {} mg / vs skutecna davka {} mg'.format(input[0], input[1], output[0], davka[i]))



#visualize interpolation
num = 100
hm_lin = np.zeros((num, num))
vek_lin = np.zeros((num, num))
for i in range(num):
    hm_lin[i] = np.linspace(4,70, num=num)  # x1
    vek_lin[i] = np.linspace(0,200, num=num)  # x2 v mesicich 1 - 200 

vek_lin = np.transpose(vek_lin)

davka_odhad = np.zeros((num, num))
for i in range(num):
    for j in range(num):
        # minmaxnorm
        x1 = (hm_lin[i, j] - x1_min)/(x1_max - x1_min)
        x2 = (vek_lin[i, j] - x2_min)/(x2_max - x2_min)
        output = forward_propagate([x1, x2])
        # un norm
        davka_odhad[i, j] = output * (y_max - y_min) + y_min


fig = plt.figure(figsize=(16,10))
fig.suptitle('Trenink modelu MLP pro predikici davkovani Panadolu \n' + 
             r'$\mu$ = ' + str(mu) + '\n' +
             'epochs = ' + str(epochs))
             
plt.subplots_adjust(hspace=0.5)

plt.subplot2grid((4,2),(0,0))
plt.plot(e_all, label='e')
plt.xlabel('epoch * k')
plt.legend()
plt.grid()

w_all_flat = []
for k in range(epochs * N):
    w_flat = []
    for i in range(len(layers) - 1):
        for r in range(layers[i]):
            for q in range(layers[i + 1]):
                w_flat.append(w_all[k][i][r][q])
    w_all_flat.append(w_flat)

plt.subplot2grid((4,2),(1,0))
plt.plot(w_all_flat)
plt.xlabel('epoch * k')
leg = []
for i in range(len(w_all_flat)):
    leg.append('w{}'.format(i)) 
plt.legend(leg)
plt.grid()

plt.subplot2grid((4,2),(2,0))
plt.plot(SSE, label='Q = e*e')
plt.xlabel('epoch \n Q_epoch_' + str(epochs) + ' = ' + str(round(SSE[epochs-1], 5)))
plt.legend()
plt.grid()

plt.subplot2grid((4,2),(3,0))
plt.plot(y, 'k', label='y_real')
plt.plot(yn, 'g', label='y_mlp')
plt.xlabel('k, epoch = ' + str(epochs))
plt.legend()
plt.grid()

ax = plt.subplot2grid((4,2),(0,1), rowspan=2)
imgplot = plt.imshow(img)
plt.title('Vizualizace struktury pouzite MLP site')
plt.axis('off')

ax = plt.subplot2grid((4,2),(2,1), rowspan=2, projection='3d')
ax.plot_surface(hm_lin, vek_lin, davka_odhad, cmap='viridis', edgecolor='none', linewidth=0, antialiased=False)
# ax.set_title('extrapolovane hodnoty davek panadolu dle veku a hmotnosti z natrenovaneho LNU\n' + 
#              'w0 = ' + str(round(w[0], 3)) + '\n' +
#              'w1 = ' + str(round(w[1], 3)) + '\n' +
#              'w2 = ' + str(round(w[2], 3)) + '\n' +
#              'davka = w0 + w1 * hmotnost + w2 * vek')
plt.title('Extrapolace davek panadolu pomoci MLP Neuronove site')
ax.set_xlabel('hmostnost [kg]')
ax.set_ylabel('vek [mesic]')
ax.set_zlabel('davka [mg]')



plt.show()

