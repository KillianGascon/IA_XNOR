
# Réseau de neurones à deux couches avec NumPy

import numpy as np

# X = entrée de nos 3 portes d'entrée XNOR
# définit les entrées du réseau de neurones
X = np.array(([0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=float)

# y = sortie du réseau de neurones
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

# Valeur que l'on peut prédire
xPredicted = np.array(([0,0,1]), dtype=float)

X = X/np.amax(X, axis=0)

# Maximum de xPredicted (données d'entrée de la prédiction
xPredicted = xPredicted/np.array(xPredicted, axis=0)

# Fichier de perte pour le graphique
lossFile = open("SumSquaredLossList.csv", "w")

class Neural_Network:
    def __init__(self):
        # Paramètres
        self.inputLayerSize = 3
        self.hiddenLayerSize = 4
        self.outputLayerSize = 1

        # Poid de chaque couche
        # On définit des valeurs aléatoires
        # Voir le diagramme d'interconnexion pour comprendre la logique

        # Matrice 3x4 pour l'entrée à cacher
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)

        # Matrice 4x1 pour la couche cachée en sortie
        self.W2 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)

    def feedForward(self, X):
        # Propagation avant traversant le réseau
        # Produit matriciel de X (entrée) et du premier ensemble de 3 x 4 poids
        self.z = np.dot(X, self.W1)

        # fonction d'activation sigmoïde (magie du réseau de neurones)
        self.z2 = self.activationSigmoid(self.z)

        # Produit matriciel de la couche cachée (z2) et du deuxième ensemble
        # de 4 x 1 poids
        self.z3 = np.dot(self.z2, self.W2)

        # Fonction d'activation finale
        o = self.activationSigmoid(self.z3)
        return o

    def BackwardPropagate(self, X, y, o):

        # Propagation arrière traversant le réseau
        # Calcule l'erreur dans la sortie
        self.o_error = y - o

        # Applique la dérivée de activationSigmoid à l'erreur
        self.o_delta = self.o_error * self.activationSigmoidPrime(o)

        # erreur z2: dans quelle mesire les poids de la couche cachée ont
        # contribué a la sortie
        # erreur
        self.z2_error = self.o_delta.dot(self.W2.T)

        # Applique la dérivée de activationSigmoid à l'erreur z2
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)

        # Ajuste les poids du premier ensemble (input layer --> hidden layer)
        self.W1 += X.T.dot(self.z2_delta)
        # Ajuste les poids du deuxième ensemble (hidden layer --> output layer)
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):

        # Boucle de rétropropagation avant
        o = self.feedForward(X)
        # Puis boucle de rétropropagation arrière sur les valeurs (feedback)
        self.BackwardPropagate(X, y, o)

    def activationSigmoid(self, s):
        # Fonction d'activation
        # Simple Courbe d'activation sigmoïde
        return 1/(1+np.exp(-s))

    def activationSigmoidPrime(self, s):
        # Première dérivée de activationSigmoid
        # On calcule la durée
        return s * (1 - s)

    def SaveSumSquaredLossList(self, i, error):
        lossFile.write(str(i) + "," + str(error.tolist()) + "\n")

    def saveWeights(self):
        # On enregistre afin de pouvoir reproduire notre réseau
        np.savetxt("WeightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("WeightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self):
        print("Predicted XOR Output data based on trained weights")
        print("Expected (X1-X3): \n" + str(xPredicted))
        print("Output (Y1): \n" + str(self.feedForward(xPredicted)))


myNeuralNetwork = Neural_Network()
trainingEpochs = 1000
# trainingEpochs = 10000

for i in range(trainingEpochs):
    print("Epoch #" + str(i) + "\n")
    print("Network Input: \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print("Actual Output of XOR Gate Neural Network: \n" + str(myNeuralNetwork.feedForward(X)))

    # Moyenne de la somme des pertes au carré
    Loss = np.mean(np.square(y - myNeuralNetwork.feedForward(X)))
    myNeuralNetwork.SaveSumSquaredLossList(i, Loss)

    print("Sum SquaredLoss: \n" + str(Loss))
    print("\n")
    myNeuralNetwork.trainNetwork(X, y)

myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput()