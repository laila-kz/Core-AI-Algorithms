#1- Encode the adn :
def decode_adn(adn):
    # Exemple simple
    n_clusters = int(adn[:3], 2) % 5 + 2   # [2,6]
    m_bits = int(adn[3:], 2)
    m = 1.5 + (m_bits / (2**(len(adn)-3))) * 1.5  # [1.5,3]

    return n_clusters, m


#2- fitness function :
def fitness_factory(X):
    def fitness(adn):
        n_clusters, m = decode_adn(adn)

        try:
            model = FCM(n_clusters=n_clusters, m=m, max_iterations=50)
            model.fit(X)

            U = model.U
            centroids = model.centroids

            pc = partition_coefficient(U)
            pe = partition_entropy(U)
            xb = xie_beni(X, U, centroids, m)
            fs = fukuyama_sugeno(X, U, centroids, m)

            # combinaison (à ajuster selon ton besoin)
            score = (
                pc
                - pe
                - xb
                - 0.001 * fs   # FS peut être grand → on réduit son impact
            )

            return score

        except:
            return -1e9  # pénalité si erreur

    return fitness


# run the thing
# X = ton dataset (numpy array)

fitness = fitness_factory(X)

ga = AlgorithmeGenetique(
    fonction_fitness=fitness,
    taille_adn=10,  # 3 bits clusters + 7 bits m
    taille_population=30,
    taux_mutation=0.05,
    max_generations=50
)

best_adn, best_score = ga.run()

print("Meilleur ADN:", best_adn)
print("Score:", best_score)

best_params = decode_adn(best_adn)
print("Meilleurs paramètres:", best_params)