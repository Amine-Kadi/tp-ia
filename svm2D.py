import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import svm



#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, C):
	"""
		Programme pour les SVM linéaires (lancé avec ESPACE)

		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points et C = ", C)

	model = svm.LinearSVC(C=C)
	model.fit(Xapp,Yapp)

	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1

	ypred = model.predict(Xtest)
	Xbleu = Xtest[ypred == 1, :]
	Xrouge = Xtest [ypred == 2, :]

	w = model.coef_[0]
	print("w = " %w)
	b = model.intercept_
	print("b = " %b)

#    yf = -w[0]/w[1] * r1 -b /w[1]
#
#    plt.plot(r1,yf,'-g')
#
#    # Pas tout à fait certain des formules
#    yfr = -w[0]/w[1]*r1-(b-1)/w[1]
#    plt.plot(r1,yfr,'-r')
#    yfb = -w[0]/w[1]*r1-(b+1)/w[1]
#    plt.plot(r1,yfb,'-b')


	# calculer et afficher la marge Delta...

	# pour réellement mettre à jour le graphique: (à garder en fin de fonction)
	fig.canvas.draw()


def monprogrammeNL(Xapp, Yapp, C, sigma):
	"""
		Programme pour les SVM non linéaires (lancé avec N)

		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
		sigma : paramètre réglé par CTRL +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points, C = ", C, " et sigma = ", sigma )

	model = svm.SVC(C=C, kernel='rbf', gamma=1/(2*sigma**2))
	model.fit(Xapp,Yapp)

	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1

	# Prédire la catégorie pour tous les points de test...

	ypred = model.predict(Xtest)
	Xbleu = Xtest[ypred == 1, :]
	Xrouge = Xtest [ypred == 2, :]

	plt.plot(Xbleu[:,0],Xbleu[:,1],'ob',alpha=0.2)

	plt.plot(Xbleu[:,0],Xbleu[:,1],'or',alpha=0.2)


	# pour réellement mettre à jour le graphique:
	fig.canvas.draw()
	print(model.n_support_)



##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
Yplot = np.zeros(0)
plotvariance = 0

C = 1
sigma = 1

def onclick(event):
	global Xplot
	global Yplot


	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.b')
		newY = np.ones(len(newX)) * 1
	elif event.button == 3 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.r')
		newY = np.ones(len(newX)) * 2
	Yplot = np.concatenate((Yplot,newY))

	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global C
	global sigma
	if event.key == " ":
		monprogramme(Xplot, Yplot, C)
	elif event.key == "n":
		monprogrammeNL(Xplot, Yplot, C, sigma)
	elif event.key == "+":
		C *= 2
		print("C = " , C)
	elif event.key == "-":
		C /= 2
		print("C = " , C)
	elif event.key == "ctrl++":
		sigma *= 2
		print("sigma = " , sigma)
	elif event.key == "ctrl+-":
		sigma /= 2
		print("sigma = " , sigma)
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points rouges")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,C)")
print("    avec la valeur de C modifiée par +/-")
print("\n N pour lancer la fonction monprogrammeNL(Xapp,Yapp,C,sigma)")
print("    avec la valeur de C modifiée par +/-")
print("    et celle de sigma modifiée par CTRL +/-\n\n")

plt.show()
