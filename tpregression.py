import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics

def kernel(X1,X2,sigma):
	"""
		Retourne la matrice de noyau K telle que K_ij = K(X1[i], X2[j])
		avec un noyau gaussien K(x,x') = exp(-||x-x'||^2 / 2sigma^2)
	"""
	m1 = X1.shape[0]
	m2 = X2.shape[0]
	K = np.zeros((m1,m2))
	for i in range(m1):
		for j in range(m2):
			K[i,j] = math.exp(- np.linalg.norm(X1[i] - X2[j])**2 / (2*sigma**2))
	return K

def krrapp(X,Y,Lambda,sigma):
	"""
		Retourne le vecteur beta du modèle Kernel ridge regression
		à noyau gaussien
		à partir d'une base d'apprentissage X,Y
	"""
	I = np.identity(len(X))
	K = kernel(X,X,sigma)
    # Pas certain
	beta = np.linalg.solve(K+(Lambda*I),Y)
	return beta

def krrpred(Xtest,Xapp,beta,sigma):
	"""
		Retourne le vecteur des prédictions du modèle
		KRR à noyau gaussien de paramètres beta et sigma
	"""
	Ktest = kernel(Xtest,Xapp,sigma)
	ypred = Ktest.dot(beta)
	return ypred


def kppvreg(Xtest, Xapp, Yapp, K):
	n = Xtest.shape[0]  # nb de points de test
	m = Xapp.shape[0]   # nb de points d'apprentissage
	ypred = np.zeros(n)

	for i in range(n):
		dist = np.zeros(m)
		for j in range(m):
			dist[j] += np.linalg.norm(Xtest[i]-Xapp[j])

		classes = Yapp[np.argsort(dist)]
		x =  (classes[:K]+1)*100000
		xx = np.bincount((x.astype(int))).argmax()

		ypred[i] += (xx/100000)-1

	return ypred

#################################################
#### Programme principal ########################
#################################################

m = 1000
X = 6 * np.random.rand(m) - 3
Y = np.sinc(X) + 0.2 * np.random.randn(m)


indexes = np.random.permutation(m)  # permutation aléatoire des 1000 indices entre 0 et 1000
indexes_app = indexes[:30]  # 30 premiers indices
indexes_test = indexes[30:] # le reste

Xapp = X[indexes_app]
Yapp = Y[indexes_app]

Xtest = X[indexes_test]
Ytest = Y[indexes_test]

# ordronner les Xtest pour faciliter le tracé des courbes

idx = np.argsort(Xtest)
Xtest = Xtest[idx]
Ytest = Ytest[idx]

# tracer la figure

plt.figure()
plt.plot(Xapp,Yapp,'*b')
plt.plot(Xtest,np.sinc(Xtest) , 'g')


Lambda = 0.3
sigma = 0.6

beta = krrapp(Xapp,Yapp,Lambda,sigma)

ypredtest = krrpred(Xtest,Xapp,beta,sigma)

ypredapp = krrpred(Xapp,Xapp,beta,sigma)

err_test = np.sqrt(metrics.mean_squared_error(Ytest, ypredtest))
err_app = np.sqrt(metrics.mean_squared_error(Yapp, ypredapp))

model = KernelRidge(alpha = Lambda, kernel='rbf', gamma = 1/(2*sigma*sigma))
model.fit(Xapp.reshape(-1,1),Yapp)

ypredtest2 = model.predict(Xtest.reshape(-1, 1))
ypredapp2 = model.predict(Xapp.reshape(-1, 1))

err_test2 = np.sqrt(metrics.mean_squared_error(Ytest, ypredtest2))
err_app2 = np.sqrt(metrics.mean_squared_error(Yapp, ypredapp2))

model_kppv = neighbors.KNeighborsRegressor(3)
model_kppv.fit(Xapp.reshape(-1,1),Yapp)

ypredtest3 = model_kppv.predict(Xtest.reshape(-1, 1))
ypredapp3 = model_kppv.predict(Xapp.reshape(-1, 1))
ypredtest4 = kppvreg(Xtest.reshape(-1, 1), Xapp.reshape(-1,1), Yapp, 2)
ypredapp4 = kppvreg(Xapp.reshape(-1, 1), Xapp.reshape(-1,1), Yapp, 2)

err_test4 = np.sqrt(metrics.mean_squared_error(Ytest, ypredtest4))
err_app4 = np.sqrt(metrics.mean_squared_error(Yapp, ypredapp4))

plt.plot(Xtest,ypredtest,'m')
plt.plot(Xtest,ypredtest2,'y')
plt.plot(Xtest,ypredtest4,'k')
plt.plot(Xtest,ypredtest3,'r')

plt.legend(['base app', 'sinc', 'Kernel ridge','Kernel ridge sklearn', 'KppRegressor Implemente','Kppv Regressor Sklearn'],bbox_to_anchor=(1.05, 1))

# Affichage des graphiques :
# (à ne faire qu'en fin de programme)
plt.show() # affiche les plots et bloque en attendant la fermeture de la fenêtre
