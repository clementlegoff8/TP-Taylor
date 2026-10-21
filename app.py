#Imports :

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

#Fonctions :

def taylor1(X, derivatives):
    Y = np.zeros_like(X)
    for k in range(len(derivatives)):
        Y_k = derivatives[k]*X**k/factorial(k)
        Y = np.add(Y,Y_k)
    return Y

def taylor2(X, f, n):
    derivees = []
    for i in range(n+1):
        derivees.append(f(0.))
        f = grad(f)
    return taylor1(X,derivees)

def custom(X):
    return 2 * np.sin(X) + np.cos(X/4)

def taux_coincidence(Y,Y_t,seuil):
    proches = abs(Y-Y_t) < seuil
    return proches.sum()/len(proches)

#Dictionnaire de fonctions :

FONCTIONS = {"sin" : np.sin, "cos" : np.cos, "exp" : np.exp, "log" : lambda x : np.log(1+x),"custom" : custom}

#Corps du site :

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.subheader("L'objectif est d'approcher une fonction par un polynôme, donné par la formule de Taylor suivante :") #sous-titre qui présente l'objectif
st.latex(r'''P_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(0)}{k!}x^{k}''') #formule latex de P_n

#Quand on ajoute :

#valeur = st.slider("Choisissez une valeur", 0, 10)
#st.write("Vous avez choisi :", valeur)

#...le scrit est réexécuté à chaque nouvelle valeur choisie sur le slider

with st.sidebar:
    choix = st.selectbox('Fonction', list(FONCTIONS.keys()))
    n = st.slider('Degré', 0, 10, value=3)
    largeur = st.slider('Largeur', 2., 20., value=12.6)
    precision = st.slider('Seuil de précision : 10^...',-6.,-2., value=-3.)
    erreur = st.checkbox("Afficher l'erreur")

f = FONCTIONS[choix]
X = np.linspace(-largeur/2,largeur/2,400)
Y = f(X)
Y_taylor = taylor2(X,f,n)

fig, ax = plt.subplots()
ax.plot(X, Y, label="Vraie fonction")
ax.plot(X, Y_taylor, label="Approximation")
ax.legend()
st.pyplot(fig)

#Pour n fixé, la qualité de l'approximation est moins bonne plus on s'éloigne de 0.

st.write(f"La proportion du domaine où les 2 fonctions sont proches (à la précision choisie) est {100*taux_coincidence(Y,Y_taylor,10**precision)} %.")

#Pour log, la comparaison avec nan ne compte pas, donc toute la partie des x < 1 diminue le taux

if erreur: #si la case "Afficher l'erreur" est cochée
    fig2, ax2 = plt.subplots()
    ax2.plot(X, abs(Y-Y_taylor))
    plt.yscale('log')
    st.pyplot(fig2)

#Comparaison de plusieurs ordres :

ajout = st.button("Ajouter cet ordre à la comparaison")
rei = st.button("Réinitialiser")

if "ordres" not in st.session_state:
    st.session_state.ordres = []

if ajout:
    st.session_state.ordres.append(Y_taylor)

if rei:
    st.session_state.ordres = []

st.write("Tous les ordres sélectionnés :")

fig3, ax3 = plt.subplots()
ax3.plot(X,Y,color="black",linewidth=2,label="Vraie fonction")
for Z in st.session_state.ordres:
    ax3.plot(X,Z)
ax3.legend()
st.pyplot(fig3)