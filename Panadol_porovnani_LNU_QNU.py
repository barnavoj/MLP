import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



hm = x1 = np.array([5.5,  7.5,  9.5,  12,  15,  18.5, 23, 29.5, 36])
vek = x2 = np.array([4.5, 6, 9, 1.5*12, 2.5*12, 4.5*12,  9*12,  13.5*12,   15*12])  # vek v mesicich
davka = y = np.array([3,    4,    5,    6,   8,   10,   13, 16,   20])

# z-score
mean_x1 = np.mean(x1)
mean_x2 = np.mean(x2)
std_x1 = np.std(x1)
std_x2 = np.std(x2)
mean_y = np.mean(y)
std_y = np.std(y)

x1 = (x1 - mean_x1)/(3*std_x1)
x2 = (x2 - mean_x2)/(3*std_x2)
y = (y - mean_y)/(3*std_y)


#---------------------------LNU GD-------------------------------

#priprava matice X
N = len(x1)
X = np.transpose(np.array([np.ones(N), x1, x2]))

# LNU
# yn = w0*x0 + w1*x1 + w2*x2
epochs = 30
mu = 0.1
nw = 3

yn = np.zeros(N)
e = np.zeros(N)
w = [0.55, -0.34, 0.24] # np.random.randn(nw)/2
dw = np.zeros(nw)
w_all = np.zeros((epochs * N, nw)) 
e_all = np.zeros((epochs * N)) 
Q = np.zeros(epochs)
kk = 0

for epoch in range(epochs):
    for k in range(N):
        yn[k] = np.dot(w, X[k])
        e[k] = y[k] - yn[k]
        # uceni - sample by sample (pro kazdou epochu j updatnu vahy k krat)
        dw = mu * e[k] * X[k]
        w = w + dw
        e_all[kk] = e[k]
        w_all[kk, :] = w
        kk += 1
    Q[epoch] = np.dot(e, e) 


fig = plt.figure(figsize=(16,10))

plt.subplot2grid((4,3),(0,0))
plt.title('LNU krokove uceni Gradient descent')
plt.plot(y, 'k', label='y_real')
plt.plot(yn, 'g', label='y_lnu')
plt.xlabel('k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(1,0))
plt.plot(e_all, label='e')
plt.xlabel('epoch * k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(2,0))
plt.plot(w_all, label=['w0','w1','w2'])
plt.xlabel('epoch * k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(3,0))
plt.plot(Q, label='Q = e*e')
plt.xlabel('epoch \n\n Q_epoch_' + str(epochs) + ' = ' + str(round(Q[epochs-1], 5)))
plt.legend()
plt.grid()

plt.subplots_adjust(hspace=0.5)



#---------------------------LNU L-M-------------------------------
#priprava matice X
N = len(x1)
X = np.transpose(np.array([np.ones(N), x1, x2]))

# LNU
# yn = w0*x0 + w1*x1 + w2*x2
E = np.eye(3)
yn = np.zeros(N)
e = np.zeros(N)
w = [0.55, -0.34, 0.24]  # np.random.randn(nw)/2
Q = np.zeros(epochs + 1)
w_all = np.zeros((epochs, 3))
e_all = np.zeros((epochs * N))
kk = 0


for epoch in range(epochs):
    yn = np.dot(X, w)
    e = y - yn
    Q[epoch] = np.dot(e, e) 
    J = X
    # uceni - batchove
    dw = (np.linalg.inv(J.T @ J + E * 1 / mu)) @ J.T @ e
    w = w + dw
    w_all[epoch, :] = w
    for i in range(N):
        e_all[kk] = e[i]
        kk += 1

yn = np.dot(X, w)
e = y - yn
Q[-1] = np.dot(e, e) 

plt.subplot2grid((4,3),(0,1))
plt.title('LNU batchove uceni Levenberg-Marquardt')
plt.plot(y, 'k', label='y_real')
plt.plot(yn, 'g', label='y_lnu')
plt.xlabel('k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(1,1))
plt.plot(e_all, label='e')
plt.xlabel('epoch * k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(2,1))
plt.plot(w_all, label=['w0','w1','w2'])
plt.xlabel('epoch')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(3,1))
plt.plot(Q, label='Q = e*e')
plt.xlabel('epoch \n\n Q_epoch_' + str(epochs) + ' = ' + str(round(Q[epochs], 5)))
plt.legend()
plt.grid()

plt.subplots_adjust(hspace=0.5)


#---------------------------QNU L-M-------------------------------
nw = 6

#priprava matice X
N = len(x1)
X = np.transpose(np.array([np.ones(N), x1, x2]))

E = np.eye(nw)
yn = np.zeros(N)
e = np.zeros(N)
w = np.random.randn(nw)/10
SSE = np.zeros(epochs + 1)
w_all = np.zeros((epochs, nw))
e_all = np.zeros((epochs * N))

J = np.zeros((N, nw))

# uceni 
kk = 0
for epoch in range(epochs):
    for k in range(N):
        #vzorec yn = suma suma xj*xi*wij
        yn[k] = 0
        for j in range(3):
            for i in range(j, 3):
                yn[k] += X[k][i] * X[k][j] * w[i + j]
                #jakobian 1. az 9. prvek
                J[k][i + j] = X[k][i] * X[k][j]

    e = y - yn
    SSE[epoch] = np.dot(e, e) 
    dw = (np.linalg.inv(J.T @ J + E * 1 / mu)) @ J.T @ e
    w = w + dw
    w_all[epoch, :] = w
    for i in range(N):
        e_all[kk] = e[i]
        kk += 1

for k in range(N):
    #vzorec yn = suma suma xj*xi*wij
    yn[k] = 0
    for j in range(3):
        for i in range(j, 3):
            yn[k] += X[k][i] * X[k][j] * w[i + j]
e = y - yn
SSE[-1] = np.dot(e, e) 


plt.subplot2grid((4,3),(0,2))
plt.title('QNU batchove uceni Levenberg-Marquardt')
plt.plot(y, 'k', label='y_real')
plt.plot(yn, 'g', label='y_qnu')
plt.xlabel('k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(1,2))
plt.plot(e_all, label='e')
plt.xlabel('epoch * k')
plt.legend()
plt.grid()

plt.subplot2grid((4,3),(2,2))
plt.plot(w_all, label=['w00','w01','w02','w11','w12','w22'])
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.grid()

plt.subplot2grid((4,3),(3,2))
plt.plot(SSE, label='Q = e*e')
plt.xlabel('epoch \n\n Q_epoch_' + str(epochs) + ' = ' + str(round(Q[epochs-1], 5)))
plt.legend()
plt.grid()

plt.subplots_adjust(hspace=0.5)


fig.suptitle('Trenink modelu pro predikici davkovani Panadolu \n' + 
             r'$\mu$ = ' + str(mu) + '\n' +
             'epochs = ' + str(epochs))


plt.show()