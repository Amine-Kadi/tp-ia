import time
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn import svm


# charger les données de défauts de rails
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1]  # tout sauf dernière colonne
y = data[:,-1]   # uniquement dernière colonne

a = 0
score = 0
scoremax = 0

G = np.zeros((1,4))
models = []

biais = 0
tab_err= np.zeros((1,4))

models = []

tps1 = time.clock()
c = 5

for j in range (140):

    X_i = np.delete(X, j, axis=0)
    y_i = np.delete(y, j)

    models = []
    for i in range (4):

        yk = 2*(y_i == i+1)-1
        yk2 = 2*(y == i+1)-1

        model = svm.LinearSVC(C=c)
        model.fit(X_i,yk)

        models.append(model)

        ypred = model.predict(X=X[j].reshape(1, -1))
        tab_err[:, i] += ypred != yk2[j]

    for k in range(4):

        G[:, k] = models[k].decision_function(X=X[j].reshape(1, -1))

    y_pred_multiclasse = np.argmax(G)+1

    biais += y_pred_multiclasse != y[j]

time2 = time.clock()
duree = time2-time1
print('time :%f' %duree, "s")

taux_err = (biais*100)/140
taux_err_pourcent = (tab_err*100)/140
print('taux err : %f'%taux_err )
print('taux err pourcent %f :'%taux_err_pourcent)
