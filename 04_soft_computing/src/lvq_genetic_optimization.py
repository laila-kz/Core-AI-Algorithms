def decode_adn_lvq(adn):
    lr_bits = int(adn[:4], 2)
    proto_bits = int(adn[4:7], 2)
    epoch_bits = int(adn[7:], 2)

    learning_rate = 0.001 + (lr_bits / 15) * 0.1
    n_prototypes = proto_bits % 5 + 1
    epochs = 50 + (epoch_bits / (2**(len(adn)-7))) * 150

    return {
        "learning_rate": learning_rate,
        "n_prototypes_per_class": n_prototypes,
        "epochs": int(epochs)
    }

def fitness_factory_lvq(X, y):
    def fitness(adn):
        params = decode_adn_lvq(adn)

        try:
            model = LVQ(**params)
            model.fit(X, y)

            qe = quantization_error(X, model)        # min
            acc = accuracy_score(X, y, model)        # max
            sil = fuzzy_silhouette(X, model)         # max
            stab = prototype_stability(X, y, params) # min

            score = (
                acc
                + 0.5 * sil
                - qe
                - 0.1 * stab
            )

            return score

        except:
            return -1e9

    return fitness



fitness = fitness_factory_lvq(X, y)

ga = AlgorithmeGenetique(
    fonction_fitness=fitness,
    taille_adn=12,
    taille_population=30,
    taux_mutation=0.05,
    max_generations=50
)

best_adn, best_score = ga.run()

print("Best ADN:", best_adn)
print("Score:", best_score)

best_params = decode_adn_lvq(best_adn)
print("Best params:", best_params)