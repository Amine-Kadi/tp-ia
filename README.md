# Intelligence artificielle & Apprentissage
## Téléchargement du code et utilisation

Vous pouvez récupérer tous les codes et explication avec cette commade :
`git clone https://github.com/Amine-Kadi/tp-ia`

Une fois le dépot récupérée, vous devez installer les dépendances nécéssaires au fonctionnement du code.
Pour ce faire, il est conseillé d'utiliser un environnement virtuel en Python.
`python -m venv tp-ia && source bin/activate && pip install -r requirements.txt`

Vous pourrez alors exécuter sans problème chaque script Python !

## TP discrimination multi-classes : Reconnaissance de caractères manuscrits par l’algorithme deskplus proches voisins

Le but de ce TP est de mettre en application l'algorithme des K plus proches voisins grâce à son implémentation dans la librairie Sci-Kit Learn en Python.

### Classification binaire linéairement séparable

Un premier type de classification est la classification *binaire* linéairement séparable.
Cela signifie dans un plan qu'il *deux* ensembles de points qui pourront être déterminés et séparés.

La classe d'un ensemble est déterminée de la manière suivate :
- On dispose d'une base de test
- On choisit un entier K qui correspond au nombre de voisin
- Pour chaque point du plan on détermine les K points voisins les plus proches dans la base de test
- En fonction du nombre de voisin d'une classe, on choisi dans quelle classe le point à classer se trouve.

Voici le code Python qui permet de créer un modèle d'apprentissage en utilisant l'algorithme des K plus proches voisins
```
	model = neighbors.KNeighborsClassifier(K)
	model.fit(Xapp,Yapp)
```
La variable `model` correspond à notre modèle d'apprentissage.
`Xapp` et `Yapp` correspondent aux bases d'apprentissage.

On effectue la prédiction avec cette ligne `ypred = model.predict(Xtest)`
Où `Xtest` correspond à la base de test.

Nous garderons ces conventions de nommages pour les prochains cas.

On peut remarquer que la frontière de séparation des classes se rapproche d'une droite.

En utilisant des variances très différentes on constate des courbures plus accentuées dans la séparation.

### Cas non séparable

Dans la réalité, il est très probable que les points ne soient pas linéairement séparable.
On se pose alors la question de l'influence du nombre de voisin pour classer des points dans des cas non séparables.

Plus on choisit de voisins, plus on a de chances de prendre des voisins éloignés et donc des voisins qui ne correspondent pas à la bonne classe.
Ainsi, en augmentant le nombre de voisin, on augmente les erreurs de classification.
Cependant, augmenter le nombre de voisin permet de réduire la taille des ensembles losqu'on a des points très concentré à un endroit du plan.

Ainsi, il s'avère important de sélectionner un bon nombre de voisin : ni trop élevé, ni trop faible, en fonction du nombre de points dans la base de test et de leur dispersion.

### Problèmes multi-classes

Un avantage de la classification avec l'algorithme des K plus proches voisins est sa capacité à résoudre des problèmes multi-classes.
Il s'agit de problèmes où la classification n'est pas binaire et linéairement séparable.

Les facteurs d'influence sur le modèle (Hyper-paramètre K et le nombre de points dans la base de test) permettent d'arriver à des conclusions similaires que pour un cas binaire.

###  Application à la reconnaissance de caractères : Base de données MNIST

L'algorithme des K plus proches voisins peut être utilisé pour des application de reconnaissances de caractères. Il s'agira dans cette partie du TP de déterminer les classes des caractères de la base de données Mnist.

La base de données Mnist est composée de 10000 exemples de tests qui correspondent à des images de 28*28 pixels représentés par des niveaux de gris, donc 784 pixel.
En niveau de gris, un pixel correspond à un octet.

La taille du modèle correspond donc à 784*10000 octets, ce qui correspond à environ 7Mo en mémoire.

Nous nous intéressons au risque du classifieur, aux erreurs de classification et à la durée de résolution.
Une erreur de classification correspond à une différence entre une valeur d'une prédiction par le modèle et la valeur réelle dans la base de test.

On détermine l'hypothétique meilleur hyper paramètre K en comparant le nombre d'erreurs obtenues pour chaque valeur.
Ainsi, pour plusieurs K, on bouclera sur une création d'un modèle d'apprentissage.

On constate que les chiffres mal classés sont des chiffres qui peuvent poser des problèmes de dinstinction pour un humain. Il s'agit par exemple majoritairement des 7 qui peuvent être confondu avec des 1. Moi-même, après avoir retiré mes lunettes, j'ai pu en confondre.



## TP : classification SVM, méthodes de décomposition et classification de défauts de rails

Le but de ce deuxième TP est d'étudier l'algorithme SVM (Machine à vecteurs supports). Pour se faire, nous allons également utiliser son implémentation dans la librairie Sci-Kit Learn.
L'algorithme SVM a pour but de déterminer des hyperplans de séparation.

### Cas séparable

Dans le cas de données linéairement séparable il faut déterminer une droite qui permettent de séparer les données.

Afin de créer un modèle d'apprentissage en utilisant l'algorithme, la méthode sera similaire que précédemment :
```
model = svm.LinearSVC(C=C)
model.fit(Xapp,Yapp)
```

On obtiendra alors un vecteur normal W : `model.coef_[0]`
Ainsi qu'un biais b : `model.intercept`

W et b nous permettent alors de déterminer la droite qui sépare les classes.
On a : ∆ = transp(T)*X+b où X correspond au vecteur support.
∆ est appelée la marge.
Afin d'affiner le résultat, on peut aussi déterminer les frontières de la marges qui correspondent aux à |transp(W)X+b|= 1 Dans ce cas, on parle de marges dures.
L'hyperplan de séparation est dit optimal.

### Cas non séparable

Comme dans le TP précédent, il est possible d'être dans un cas non séparable à classer.
Dans ce cas, l'hyperplan de séparation optimal n'est plus défini.

Il s'agit alors d'appliquer le même algorithme avec une valeur de C finie.

On constate qu'une faible valeur de C permet d'obtenir de très minces marges.

### Cas non linéaire

Dans le cas non linéaire, il faut rajouter un hyperparamètre sigma.
La frontière de séparation entre les classes n'est plus une droite, dans le plan, cela ressemblerait plus à des courbes.
Le noyau ne sera donc plus linéaire.

C et sigma ont une influence non négligeable sur le modèles et ses caractéristiques.
Un C trop faible est souvent synonyme de sur-apprentissage. Cela correspond à un problème de dimensionnement de la résolution du problème qu'il peut-être important d'optimiser pour des raisons de performances machines mais aussi et surtout pour des raisons de performance d'un point de vue du résultat. Le sur-apprentissage peut-être un facteur d'erreur. Un petit C donne beaucoup de vecteurs supports.
Un sigma trop élevée est quand à lui synonyme de sous-apprentissage et donc d'erreur à cause d'un manque de valeur d'apprentissage qui seront utilisées pour les vecteurs supports.
Un sigma trop faible permet un bon modèle sur une base d'apprentissage et une base de test. Mais en changeant de base de test, on pourrait voir apparaitre des erreurs.


### Problème multi-classe et techniques de validation croisée : application à la classification de défauts de rails

Il est également possible avec l'algorithme SVM de traiter des problèmes multi-classe
Pour l'algorithme SVM, il s'agit d'appliquer plusieurs fois des modèles avec des classifieurs binaires pour faire un choix au final.

Afin d'étudier ces problèmes, nous étudirons la classifications de défauts de rails.

###  Classifieurs binaires et combinaison de classifieurs binaires
Nous commençons par un classifieur binaire pour les 4 types de défauts.
On testes les taux d'erreurs sur les différents classifieurs.

### Estimation de l'erreur générale par validation croisée.

Dans notre cas, on peut considérer que les taux d'erreurs sont biaisés : nous avons utilisés la même base pour l'apprentissage que pour le test.
Il est important d'adopter une autre technique de validation. En l'occurence, nous utiliserons la méthode de validation croisée.
Celle-ci correspond au fait de couper la base d'apprentissage en plusieurs morceaux, de boucler sur ces morceaux et utilant un seul morceau pour la base de tests et tous les autres pour la base d'apprentissage.

La méthode permettant une meilleure estimation de l'erreur par validation croisée est la méthode LOO : leave one out.
Il s'agit de couper la base d'apprentissage en dans des morceaux de la plus petite taille possible : ainsi, on considère des morceaux de taille 1. On garde une seule valeur pour le test et toutes les autres pour l'apprentissage.
Pour chaque classifieurs, on boucle alors sur l'intégralité des valeurs pour effectuer l'estimation de l'erreur générale.

L'erreur de validation sera le nombre d'erreur au total de toute l'opération divisé par le nombre d'élément dans la base complète.
De manière très concrète : il s'agit de vérifier point par point si l'*étiquette* attribuée par le modèle déterminé sera la bonne.


Ces erreurs permettent de relativiser l'erreur du classifieur global.


## TP : régression non linéaire (et non paramétrique)

Le but de ce TP est d'étudier des cas d'apprentissage pour des problèmes de régression non linéaire.
Je pense que le cours associé à ce TP était assez compliqué à suivre et à comprendre à distance malheureusement. Ainsi, je pense que les explications seront approximatives.


### Méthodes non paramétriques pour la régression non linéaire
On construit au préalable un graphique à l'aide de d'un jeu de 1000 données que l'on sépare dans des bases d'apprentissage de 30 exemples (toutes les autres données seront utilisées pour la base de test)
Ensuite, il s'agit de tracer la fonction de régression associée aux données de la base d'apprentissage.
Pour cela, on utilise la librairie Numpy.
On peut observer le résultat ci-dessous :

Afin d'utiliser des méthodes non paramétriques pour apprendre des fonctions, il s'agira d'implémenter l'algorithme d'apprentissage de la Kernel ridge regression.


## TP apprentissage non supervisé : K-means et spectral clustering
Ce TP a pour but d'étudier un autre aspect de l'intelligence artificielle et de l'apprentissage, il s'agit du clustering.
Cela correspond à créer des clusters de données qui vont former ensemble un groupe. Rajouter un point dans une expérience permettra de détreminer dans quel groupe il appartient.

### Test de l'algorithme dans le plan et K-means
Le principe de l'algorithme k-means est de former les différents cluster en déterminant un point au centre d'un cluster.
Il s'agit de minimiser la distance euclidienne entre un point d'un cluster et le cluster. Le centre sera le point obtenu en faisant la moyennes des distances minimales.

Au départ, un centre d'un cluster aléatoire est tiré. L'algorithme rentre alors dans une boucle.
A chaque itération, un nouveau centre est calculé et va donc changer la forme du cluster. Au bout d'un nombre d'itération finie, on peut être sûr de converger vers le centre réel.
Rajouter un point permettra impliquera de recalculer un centre. Afin d'accélerer le processus, on peut considérer que le premier centre était proche de la réalité. On repart donc avec une information qui nous permettra de converger plus rapidement vers un nouveau cluster.

On constate que le tirage aléatoire est une mauvaise idée. Chaque cluster est calculé au départ avec un centre.
On calculera donc des cluster en se basant sur des points qui peuvent être très éloignés à plusieurs reprise.
Il s'avère intéressant de partir directement d'un bon point.
L'initialisation de l'algorithme consistera alors en une sélection du meilleur centre de départ.
On bouclera sur un nombre de centre aléatoire préalablement fixé de point au départ afin de déterminer le meilleur.
Le meilleur est celui qui aura la distance moyenne la plus faible.

### Spectral clustering

Le clustering spectral est une méthode de classification non supervisée qui va utiliser une mesure de la similarité entre les points.
La différence entre l'algorithme K-Means et le clustering spectrale est important : le clustering k-means se base sur une moyenne des distance des points au centre d'un cluster alors que le clustering spectrale se base sur la "connectivité" des points d'un même cluster.
En clair : le clustering par méthode spectrale peut faire penser à l'algorithme des K plus proches voisins : on va regarder les points autour d'un point particulier pour déterminer le cluster.
On parle alors d'affinité.
L'affinté peut être vu comme un indicateur qui va expliciter le "degré" d'apartenance d'un point à un cluster.
Chaque cluster va "accepter" ou "rejeter" le point.
Au final, chaque point converge vers un cluster et chaque cluster converge vers sa forme "réelle" dans un cas idéal.

On applique l'algorithme spectral clustering de la même manière que l'algorithme k-means.
Il s'agira alors de considérer la matrice d'affinité A (`affinte` dans mon code pour plus de lisibilité) déterminée avec le noyau gaussien de paramètre gamme : γ = 1/2σ^2
