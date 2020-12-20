import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

images = ['china.jpg', 'flower.jpg']

K = 2

# tests sur les 2 images
for image in images:
	img = load_sample_image(image)

	(width,height,channels) = img.shape

	# Création de la matrice des pixels
	print(img.shape)
	X = img.reshape(-1,3)
    #Appel et entraînement de Kmeans
	clustering = KMeans(n_clusters=K, init='random', n_init=100 )
	clustering.fit(X)
	y = clustering.labels_ #Prédictions
	centres = np.uint8(clustering.cluster_centers_) #Récuperations des centres

	# récupération des étiquettes sous forme matricielle :
	Y = y.reshape(-1,1) # à modifier...
	# Création de l'image couleur à 3 canaux
	segmentation = centres[Y].reshape((img.shape))  # taille identique à img

	# afficher l'originale, les étiquettes Y et la segmentation avec la couleur moyenne :
	plt.figure()
	plt.imshow(img)
	plt.draw()
	plt.figure()
	plt.imshow(segmentation)
	plt.draw()

plt.show()
