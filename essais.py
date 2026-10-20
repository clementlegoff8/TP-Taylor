import autograd.numpy as np
from autograd import grad

#2

#2.1

X = np.linspace(0,2*np.pi,100) #tableau de 100 réels régulièrement espacés entre 0 et 2π

#2.2

sin_der = grad(np.sin) #dérivée de sin
sin_der = np.vectorize(sin_der) #vectorisation

#2.3

#print(sin_der(X) - np.cos(X)) #vérification

#On a bien un tableau de 0

#3

#3.1

from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives): #approximation de Taylor sur X
    Y = np.zeros_like(X)
    for k in range(len(derivatives)):
        Y_k = derivatives[k]*X**k/factorial(k)
        Y = np.add(Y,Y_k)
    return Y

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)

#plt.plot(X,taylor1(X,sinus10))
#plt.plot(X,np.sin(X))
#plt.show()

#On a bien les deux courbe qui ont la même forme autour de 0

#3.2

def taylor2(X, f, n):
    derivees = []
    for i in range(n+1):
        derivees.append(f(0.))
        f = grad(f)
    return taylor1(X,derivees)

Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)

#plt.plot(X,Y)
#plt.plot(X,Y2)
#plt.show()

def taux_coincidence(Y,Y_t,seuil): #renvoie le pourcentage du domaine où Y et Y_t sont proches à seuil près
    proches = abs(Y-Y_t) < seuil
    return proches.sum()/len(proches)

#print(taux_coincidence(Y,Y2,10**(-3)))

#Y et Y2 sont proches à 10^-3 près sur 42% du domaine

Z = np.cos(X)
Z2 = taylor2(X, np.cos, 0)

#plt.plot(X,Z)
#plt.plot(X,Z2)
#plt.show()

#À l'ordre 0, en 0, le cosinus est juste la droite y=1

#3.3

def custom(X):
    return 2 * np.sin(X) + np.cos(X/4)

Y3 = custom(X)
Y4 = taylor2(X,custom,14)

#plt.plot(X,Y3)
#plt.plot(X,Y4)
#plt.show()

#print(taux_coincidence(Y3,Y4,10**(-3)))

#On obtient 61,5%