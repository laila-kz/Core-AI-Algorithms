import numpy as np

class Neurone:
    def __init__(self,n_entrees):
        self.poids= np.random.rand(n_entrees) * 0.1 
        self.biais=np.random.rand() * 0.1
        self.derniere_entree=None
        self.derniere_sortie=None
        self.gradients_poids=None
        self.gradient_biais=None
    def fct_activation(self, entree):
        return 1/(1+np.exp(-entree))
    
    def fct_activation_derivative(self, sortie):
        return sortie * (1 - sortie)
    
    def forward(self, entree):
        self.derniere_entree= entree
        z=np.dot(self.poids, entree) + self.biais
        self.derniere_sortie=self.fct_activation(z)
        return self.derniere_sortie
    
    def backward(self,erreur_recue , learning_rate):
        gradient_local = erreur_recue * self.fct_activation_derivative(self.derniere_sortie)
        self.gradients_poids = gradient_local * self.derniere_entree
        self.gradient_biais = gradient_local

        #compute the error to propagate to the previous layer
        erreur_precedente = gradient_local * self.poids

        #update weights and bias
        self.poids -= learning_rate * self.gradients_poids
        self.biais -= learning_rate * self.gradient_biais
        return erreur_precedente
    


class Couche:
    def __init__(self, n_neurones, n_entrees_par_neurone):
        self.neurones = [Neurone(n_entrees_par_neurone) for _ in range(n_neurones)]
    
    def forward(self, entree):
        return np.array([neurone.forward(entree) for neurone in self.neurones])
    
    def backward(self, erreurs_recues, learning_rate):
        erreurs_precedentes = np.zeros_like(self.neurones[0].derniere_entree)
        for i, neurone in enumerate(self.neurones):
            erreur_precedente = neurone.backward(erreurs_recues[i], learning_rate)
            erreurs_precedentes += erreur_precedente
        return erreurs_precedentes
    

class ReseauNeurones:
    def __init__(self, structure):
        """
        structure: liste du nombre de neurones par couche
        ex: [3, 4, 1] pour 3 entrées, 4 neurones cachés, 1 sortie
        """
        self.couches = []
        
        for i in range(len(structure) - 1):
            couche = Couche(structure[i + 1], structure[i])
            self.couches.append(couche)
    
    def forward(self, X):
        """Propagation avant à travers tout le réseau"""
        sortie = X
        for couche in self.couches:
            sortie = couche.forward(sortie)
        return sortie
    
    def backward(self, y_pred, y_true, learning_rate):
        """Rétropropagation du gradient"""
        # Calcul de l'erreur initiale (sortie - cible)
        erreur = y_pred - y_true
        
        # Propagation arrière à travers les couches
        for couche in reversed(self.couches):
            erreur = couche.backward(erreur, learning_rate)
    
    def entrainer(self, X, y, epochs, learning_rate, verbose=True):
        """Entraînement du réseau"""
        historique_erreur = []
        
        for epoch in range(epochs):
            erreur_totale = 0
            
            for i in range(len(X)):
                # Forward
                y_pred = self.forward(X[i])
                
                # Calcul de l'erreur
                erreur = np.mean((y_pred - y[i]) ** 2)
                erreur_totale += erreur
                
                # Backward
                self.backward(y_pred, y[i], learning_rate)
            
            erreur_moyenne = erreur_totale / len(X)
            historique_erreur.append(erreur_moyenne)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Erreur: {erreur_moyenne:.6f}")
        
        return historique_erreur
    
    def predire(self, X):
        """Faire des prédictions"""
        predictions = []
        for x in X:
            predictions.append(self.forward(x))
        return np.array(predictions)

# Exemple d'utilisation avec le problème XOR
print("=== Test avec le problème XOR ===")

# Données d'entraînement (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Création du réseau: 2 entrées, 4 neurones cachés, 1 sortie
reseau = ReseauNeurones([2, 4, 1])

# Entraînement
print("Entraînement en cours...")
historique = reseau.entrainer(X, y, epochs=1000, learning_rate=0.5)

# Test
print("\nPrédictions après entraînement:")
predictions = reseau.predire(X)
for i in range(len(X)):
    print(f"Entrée: {X[i]}, Sortie attendue: {y[i][0]}, Prédiction: {predictions[i][0]:.4f}")

# Arrondi pour classification
print("\nClassification (arrondi):")
for i in range(len(X)):
    classification = 1 if predictions[i][0] > 0.5 else 0
    print(f"Entrée: {X[i]}, Sortie attendue: {y[i][0]}, Prédiction: {classification}")

# Deuxième exemple: Régression simple
print("\n=== Test avec régression ===")

# Données: fonction sinus
X_reg = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_reg = np.sin(X_reg)

# Création du réseau: 1 entrée, 10 neurones cachés, 1 sortie
reseau_reg = ReseauNeurones([1, 10, 1])

# Entraînement
print("Entraînement en cours...")
reseau_reg.entrainer(X_reg, y_reg, epochs=500, learning_rate=0.1, verbose=False)

# Test sur quelques points
print("\nQuelques prédictions:")
points_test = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
predictions_reg = reseau_reg.predire(points_test)

for i in range(len(points_test)):
    print(f"x: {points_test[i][0]:.2f}, sin(x) réel: {np.sin(points_test[i][0]):.4f}, "
        f"prédiction: {predictions_reg[i][0]:.4f}")