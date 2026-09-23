import numpy as np 
import matplotlib.pyplot as plt



class NeuronRNN:
    def __init__(self, n_entrees , n_cache):
        self.poids_entree = np.random.randn(n_entrees ) * 0.1

        #poids pour les connexions récurrentes
        self.poids_cache = np.random.randn(n_cache) * 0.1

        self.biais = np.random.randn() * 0.1
        self.derniere_entree = None
        self.dernier_cache = None 
        self.derniere_entree_cache = None

    def fct_activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fct_activation_derivative(self, sortie):
        return sortie * (1 - sortie)
    
    def forward(self, entree, cache_precedent):
        self.derniere_entree = entree
        self.dernier_cache = cache_precedent

        z = np.dot(self.poids_entree, entree) + np.dot(self.poids_cache, cache_precedent) + self.biais
        sortie = self.fct_activation(z)
        return sortie
    


    def backward(self, erreur_recue, learning_rate):
        gradient_local = erreur_recue * self.fct_activation_derivative(self.dernier_cache)
        gradients_poids_entree = gradient_local * self.derniere_entree
        gradients_poids_cache = gradient_local * self.dernier_cache
        gradient_biais = gradient_local

        #compute the error to propagate to the previous time step
        erreur_precedente_cache = gradient_local * self.poids_cache

        #update weights and bias
        self.poids_entree -= learning_rate * gradients_poids_entree
        self.poids_cache -= learning_rate * gradients_poids_cache
        self.biais -= learning_rate * gradient_biais

        return erreur_precedente_cache
    


class CoucheRNN:
    def __init__(self, n_neurones, n_entrees_par_neurone, n_cache):
        self.neurones = [NeuronRNN(n_entrees_par_neurone, n_cache) for _ in range(n_neurones)]
        self.n_neurones = n_neurones

        #store the hidden states and inputs for backpropagation through tim
        self.etats_cache = []
        self.entrees = []



    def forward(self, sequence_entrees):
        sorties = []
        etat_cache = np.zeros(self.n_neurones)  # État initial (généralement zéro)
        
        self.etats_cache = [etat_cache.copy()]  # Sauvegarde pour BPTT
        self.entrees = []
        
        for t, entree in enumerate(sequence_entrees):
            self.entrees.append(entree)
            
            # Calcul de la sortie pour chaque neurone à ce pas de temps
            sortie_t = []
            nouvel_etat = []
            for neurone in self.neurones:
                sortie = neurone.forward(entree, etat_cache)
                sortie_t.append(sortie)
                nouvel_etat.append(sortie)  # L'état caché est la sortie du neurone
            sorties.append(sortie_t)
            etat_cache = np.array(nouvel_etat)
            self.etats_cache.append(etat_cache.copy())  # Sauvegarde pour BPTT
        return np.array(sorties)
    


    def backward(self, erreurs_recues, learning_rate):
        erreurs_precedentes_cache = np.zeros(self.n_neurones)
        
        # Parcours à l'envers des pas de temps
        for t in reversed(range(len(self.entrees))):
            entree_t = self.entrees[t]
            etat_cache_t = self.etats_cache[t]
            etat_cache_suivant = self.etats_cache[t + 1]

            erreurs_t = erreurs_recues[t] + erreurs_precedentes_cache
            
            # Backpropagation pour chaque neurone à ce pas de temps
            erreurs_precedentes_cache = np.zeros(self.n_neurones)
            for i, neurone in enumerate(self.neurones):
                erreur_recue = erreurs_t[i]
                erreur_precedente_cache = neurone.backward(erreur_recue, learning_rate)
                erreurs_precedentes_cache += erreur_precedente_cache


class ReseauRNN:
    def __init__(self, structure):
        
        self.taille_entree = structure[0]
        self.taille_cache = structure[1]
        self.taille_sortie = structure[2]
        
        # Couche RNN cachée
        self.couche_rnn = CoucheRNN(self.taille_cache, self.taille_entree)
        
        # Couche de sortie (feedforward simple)
        self.poids_sortie = np.random.randn(self.taille_sortie, self.taille_cache) * 0.1
        self.biais_sortie = np.random.randn(self.taille_sortie) * 0.1
        
        # Stockage pour la rétropropagation
        self.dernieres_sorties_cachees = None
    
    def fct_activation_sortie(self, x):
        # Pour la classification, on peut utiliser softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, sequence_entrees):
        """
        Traite une séquence complète
        """
        # Propagation à travers la couche RNN
        sorties_cachees = self.couche_rnn.forward(sequence_entrees)
        self.dernieres_sorties_cachees = sorties_cachees
        
        # Couche de sortie (appliquée à chaque pas de temps)
        sorties = []
        for t in range(len(sorties_cachees)):
            sortie_t = np.dot(self.poids_sortie, sorties_cachees[t]) + self.biais_sortie
            sortie_t = self.fct_activation_sortie(sortie_t)
            sorties.append(sortie_t)
        
        return np.array(sorties)
    
    def backward(self, sorties_predites, cibles, learning_rate):
        """
        Rétropropagation à travers le temps (BPTT)
        """
        T = len(sorties_predites)  # Longueur de la séquence
        
        # Gradients pour la couche de sortie
        d_poids_sortie = np.zeros_like(self.poids_sortie)
        d_biais_sortie = np.zeros_like(self.biais_sortie)
        
        # Gradients pour la couche RNN
        d_poids_entree = [np.zeros_like(n.poids_entree) for n in self.couche_rnn.neurones]
        d_poids_cache = [np.zeros_like(n.poids_cache) for n in self.couche_rnn.neurones]
        d_biais = [np.zeros_like(n.biais) for n in self.couche_rnn.neurones]
        
        # Erreur à propager dans le temps
        erreur_cache_suivant = np.zeros(self.taille_cache)
        
        # Backward à travers le temps (de t = T-1 à 0)
        for t in reversed(range(T)):
            # Gradient pour la couche de sortie
            erreur_sortie = sorties_predites[t] - cibles[t]
            d_poids_sortie += np.outer(erreur_sortie, self.dernieres_sorties_cachees[t])
            d_biais_sortie += erreur_sortie
            
            # Erreur propagée à la couche cachée
            erreur_cache = np.dot(self.poids_sortie.T, erreur_sortie) + erreur_cache_suivant
            
            # Gradient pour chaque neurone à ce pas de temps
            for i, neurone in enumerate(self.couche_rnn.neurones):
                # Dérivée de l'activation
                derivee = neurone.fct_activation_derivee(self.couche_rnn.etats_cache[t + 1][i])
                
                # Erreur locale du neurone
                erreur_locale = erreur_cache[i] * derivee
                
                # Gradients pour ce pas de temps
                d_poids_entree[i] += erreur_locale * self.couche_rnn.entrees[t]
                d_poids_cache[i] += erreur_locale * self.couche_rnn.etats_cache[t][i]
                d_biais[i] += erreur_locale
                
                # Erreur à propager au pas précédent (via poids caché)
                erreur_cache_suivant[i] = erreur_locale * neurone.poids_cache[i]
        
        # Mise à jour des poids
        # Couche de sortie
        self.poids_sortie -= learning_rate * d_poids_sortie / T
        self.biais_sortie -= learning_rate * d_biais_sortie / T
        
        # Couche RNN
        for i, neurone in enumerate(self.couche_rnn.neurones):
            neurone.poids_entree -= learning_rate * d_poids_entree[i] / T
            neurone.poids_cache -= learning_rate * d_poids_cache[i] / T
            neurone.biais -= learning_rate * d_biais[i] / T
    
    def entrainer(self, sequences, cibles, epochs, learning_rate, verbose=True):
        """
        Entraîne le RNN sur un ensemble de séquences
        """
        historique_erreur = []
        
        for epoch in range(epochs):
            erreur_totale = 0
            
            for i in range(len(sequences)):
                # Forward
                predictions = self.forward(sequences[i])
                
                # Calcul de l'erreur (cross-entropy)
                erreur = -np.sum(cibles[i] * np.log(predictions + 1e-8))
                erreur_totale += erreur
                
                # Backward
                self.backward(predictions, cibles[i], learning_rate)
            
            erreur_moyenne = erreur_totale / len(sequences)
            historique_erreur.append(erreur_moyenne)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Erreur: {erreur_moyenne:.6f}")
        
        return historique_erreur
    
    def predire(self, sequence, longueur_prediction=1):
        """
        Prédit la suite d'une séquence
        """
        predictions = []
        
        # Forward sur la séquence d'entrée
        sorties = self.forward(sequence)
        predictions.extend(sorties)
        
        # Prédiction auto-régressive (utilise ses propres prédictions comme entrée)
        dernier_etat = self.couche_rnn.etats_cache[-1]
        derniere_entree = sequence[-1]
        
        for _ in range(longueur_prediction):
            # Utilise la dernière prédiction comme prochaine entrée
            nouvelle_sortie = np.dot(self.poids_sortie, dernier_etat) + self.biais_sortie
            nouvelle_sortie = self.fct_activation_sortie(nouvelle_sortie)
            predictions.append(nouvelle_sortie)
            
            # Met à jour pour la prochaine itération
            derniere_entree = nouvelle_sortie
        
        return np.array(predictions)

# Exemple 1: Prédiction de séquence binaire simple
print("=== Test RNN avec séquence binaire ===")

# Créer une séquence simple: 0,1,0,1,0,1,...
def creer_sequence_binaire(longueur):
    X = []
    y = []
    for i in range(longueur - 1):
        # One-hot encoding pour l'entrée
        if i % 2 == 0:
            X.append([1, 0])  # [1,0] représente 0
            y.append([1, 0])  # Prédire 0
        else:
            X.append([0, 1])  # [0,1] représente 1
            y.append([0, 1])  # Prédire 1
    return np.array(X), np.array(y)

# Créer les données
X_seq, y_seq = creer_sequence_binaire(20)
X_seq = X_seq.reshape(-1, 2)  # Redimensionner pour avoir des séquences
y_seq = y_seq.reshape(-1, 2)

print(f"Forme de la séquence: {X_seq.shape}")

# Créer et entraîner le RNN
rnn = ReseauRNN([2, 5, 2])  # 2 entrées, 5 neurones cachés, 2 sorties

print("Entraînement du RNN...")
historique = rnn.entrainer([X_seq], [y_seq], epochs=500, learning_rate=0.01)

# Tester la prédiction
print("\nTest de prédiction:")
test_seq = np.array([[1, 0], [0, 1], [1, 0]])  # Séquence: 0,1,0
predictions = rnn.predire(test_seq, longueur_prediction=3)

print("Séquence d'entrée: 0,1,0")
print("Prédictions (prochaines valeurs):")
for i, pred in enumerate(predictions[len(test_seq):]):
    print(f"  Pas {i+1}: 0: {pred[0]:.4f}, 1: {pred[1]:.4f}")

# Exemple 2: Classification de séquences
print("\n=== Test RNN pour classification de séquences ===")

# Créer des séquences de différentes longueurs
sequences = [
    np.array([[1,0], [1,0], [1,0]]),  # Trois 0: classe 0
    np.array([[0,1], [0,1], [0,1]]),  # Trois 1: classe 1
    np.array([[1,0], [0,1], [1,0]]),  # Alternance: classe 0
    np.array([[0,1], [1,0], [0,1]]),  # Alternance: classe 1
]

cibles = [
    np.array([[1,0]] * 3),  # Classe 0 pour chaque pas
    np.array([[0,1]] * 3),  # Classe 1 pour chaque pas
    np.array([[1,0]] * 3),  # Classe 0 pour chaque pas
    np.array([[0,1]] * 3),  # Classe 1 pour chaque pas
]

# Créer un nouveau RNN
rnn_clf = ReseauRNN([2, 4, 2])

# Entraînement
print("Entraînement du classifieur de séquences...")
for epoch in range(300):
    erreur_epoch = 0
    for seq, cible in zip(sequences, cibles):
        pred = rnn_clf.forward(seq)
        erreur = -np.sum(cible * np.log(pred + 1e-8))
        erreur_epoch += erreur
        rnn_clf.backward(pred, cible, 0.01)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Erreur: {erreur_epoch/len(sequences):.6f}")

# Test
print("\nClassification des séquences:")
for i, seq in enumerate(sequences):
    pred = rnn_clf.forward(seq)
    classe_predite = np.argmax(pred[-1])  # Dernière prédiction
    classe_reelle = np.argmax(cibles[i][-1])
    print(f"Séquence {i+1}: Prédite={classe_predite}, Réelle={classe_reelle}")