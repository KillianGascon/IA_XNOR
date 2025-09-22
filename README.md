# Réseau de Neurones XNOR avec NumPy

Ce projet implémente un **réseau de neurones à deux couches** (1 couche cachée) en utilisant uniquement **NumPy**.  
L’objectif est d’apprendre la fonction **XNOR** à partir des 3 entrées binaires possibles.

---

## 🚀 Fonctionnalités

- Réseau de neurones simple avec :
  - 3 neurones en entrée
  - 4 neurones dans la couche cachée
  - 1 neurone en sortie
- Fonction d’activation : **sigmoïde**
- Apprentissage par **rétropropagation** (backpropagation)
- Sauvegarde des poids entraînés dans des fichiers texte
- Suivi de la **perte quadratique moyenne** (Sum Squared Loss) dans un fichier CSV
- Prédiction d’une sortie à partir d’une nouvelle entrée

---

## 📂 Structure du projet

IA_XNOR/
│── main.py # Script principal du réseau de neurones
│── SumSquaredLossList.csv # Historique de la perte (loss) par époque
│── WeightsLayer1.txt # Poids entraînés de la couche entrée → cachée
│── WeightsLayer2.txt # Poids entraînés de la couche cachée → sortie

---

## ⚙️ Installation

1. Cloner le dépôt ou copier les fichiers dans un dossier :

```bash
git clone https://github.com/ton-utilisateur/IA_XNOR.git
cd IA_XNOR
```

2. Installer Python 3.x et NumPy :
3. 
```bash
pip install numpy
```

▶️ Utilisation

Lancer l’entraînement avec :

```bash
python main.py
```
---

Pendant l’exécution :

Le réseau affiche les prédictions par époque

La perte (loss) est sauvegardée dans SumSquaredLossList.csv

Les poids finaux sont sauvegardés dans WeightsLayer1.txt et WeightsLayer2.txt

---

📊 Résultats

Après l’entraînement, le réseau doit approximer la table de vérité XNOR :

Entrées (X1, X2, X3)	Sortie attendue
0 0 0	1
0 0 1	0
0 1 0	0
0 1 1	0
1 0 0	0
1 0 1	0
1 1 0	0
1 1 1	1

Le fichier CSV SumSquaredLossList.csv peut être utilisé pour tracer la courbe d’évolution de la perte.

---

🛠 Améliorations possibles

Ajouter un taux d’apprentissage (learning rate) pour un meilleur contrôle

Implémenter d’autres fonctions d’activation (ReLU, tanh…)

Visualiser la perte avec matplotlib

Étendre le réseau à d’autres problèmes logiques ou datasets

---

👤 Auteur

Projet réalisé par Killian Gascon dans le cadre d’un apprentissage des bases des réseaux de neurones avec NumPy.
