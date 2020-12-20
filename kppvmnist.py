import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn import neighbors
import time

#### Fonctions de chargement et affichage de la base mnist ####

def load_mnist(m,mtest):

	X = np.load("mnistX.npy")
	y = np.load("mnisty.npy")

	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))

	return train_test_split(X, y, train_size=m, test_size=mtest)


def showimage(x):
	plt.imshow( 255 - np.reshape(x, (28, 28) ), cmap="gray")
	plt.show()


def kppvpredict(Xtest, Xapp,Yapp, K):

    #Création d'une matrice où va être stoqué les prédictions
    pred = np.zeros(len(Xtest))

    for i in range(len(Xtest)):
        dists = np.linalg.norm(Xtest[i]-Xapp,axis=1)
        classes = Yapp[np.argsort(dist)]
        pred[i] += np.bincount(classes[:K]).argmax()
    return pred

Xtrain, Xtest, ytrain, ytest = load_mnist(11000, 1000)

time1 = time.clock()
res = kppvpredict(Xtest,Xtrain,ytrain,3)
time2 = time.clock()
duree = time2-time1
print('duree :%f' %duree, "s")

err = res!=ytest
moy = np.mean(err)*100
print('taux err moy %f' %moy)
