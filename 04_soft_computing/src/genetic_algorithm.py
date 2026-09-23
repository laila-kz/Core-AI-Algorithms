#new darwin:
import numpy as np
import random

class AlgorithmeGenetique:
    def __init__(self, fonction_fitness, taille_adn, taille_population=100,
                 taux_mutation=0.05, max_generations=100):
        self.fonction_fitness = fonction_fitness
        self.taille_adn = taille_adn
        self.taille_population = taille_population
        self.taux_mutation = taux_mutation
        self.max_generations = max_generations

    def _initialiser_population(self):
        return ["".join(random.choice("01") for _ in range(self.taille_adn))
                for _ in range(self.taille_population)]

    def _croisement(self, parent_a, parent_b):
        point = random.randint(1, self.taille_adn - 1)
        return (
            parent_a[:point] + parent_b[point:],
            parent_b[:point] + parent_a[point:]
        )

    def _mutation(self, enfant):
        return "".join(
            "1" if (bit == "0" and random.random() < self.taux_mutation)
            else "0" if (bit == "1" and random.random() < self.taux_mutation)
            else bit
            for bit in enfant
        )

    def _diversite(self, population):
        # diversité = distance moyenne entre individus
        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        total = 0
        count = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                total += hamming(population[i], population[j])
                count += 1
        return total / (count + 1e-10)

    def run(self):
        population = self._initialiser_population()

        historique_best = []
        diversites = []

        meilleur_absolu = None
        meilleur_score = -1e9

        for gen in range(self.max_generations):

            scores = [(ind, self.fonction_fitness(ind)) for ind in population]
            scores.sort(key=lambda x: x[1], reverse=True)

            best_score_gen = scores[0][1]
            historique_best.append(best_score_gen)

            if best_score_gen > meilleur_score:
                meilleur_absolu = scores[0][0]
                meilleur_score = best_score_gen

            # diversité
            diversites.append(self._diversite(population))

            # sélection
            parents = [ind for ind, _ in scores[:self.taille_population // 2]]

            # reproduction
            new_pop = []
            while len(new_pop) < self.taille_population:
                a, b = random.sample(parents, 2)
                e1, e2 = self._croisement(a, b)
                new_pop.append(self._mutation(e1))
                if len(new_pop) < self.taille_population:
                    new_pop.append(self._mutation(e2))

            population = new_pop

        return {
            "best_adn": meilleur_absolu,
            "best_score": meilleur_score,
            "history": historique_best,
            "diversity": diversites
        }
    

#les critere de validation de darwin:
def critere_fitness(result):
    return result["best_score"]

def critere_convergence(history):
    max_score = max(history)
    threshold = 0.95 * max_score

    for i, val in enumerate(history):
        if val >= threshold:
            return 1 / (i + 1)  # plus tôt = meilleur

    return 0

def critere_diversite(diversity_history):
    return np.mean(diversity_history)


def critere_succes(ga_builder, runs=5, seuil=0.9):
    success = 0

    for _ in range(runs):
        result = ga_builder().run()
        if result["best_score"] >= seuil:
            success += 1

    return success / runs



#main part 
def decode_adn_ga(adn):
    mut_bits = int(adn[:4], 2)
    pop_bits = int(adn[4:8], 2)
    gen_bits = int(adn[8:], 2)

    mutation = 0.001 + (mut_bits / 15) * 0.1
    population = 20 + pop_bits * 5
    generations = 20 + gen_bits * 5

    return mutation, population, generations

def fitness_ga_factory(base_fitness, taille_adn_problem):

    def fitness(adn):
        mutation, population, generations = decode_adn_ga(adn)

        def build_ga():
            return AlgorithmeGenetique(
                fonction_fitness=base_fitness,
                taille_adn=taille_adn_problem,
                taille_population=population,
                taux_mutation=mutation,
                max_generations=generations
            )

        try:
            result = build_ga().run()

            f = critere_fitness(result)
            conv = critere_convergence(result["history"])
            div = critere_diversite(result["diversity"])
            succ = critere_succes(build_ga, runs=3)

            score = (
                f
                + 0.5 * conv
                + 0.3 * div
                + succ
            )

            return score

        except:
            return -1e9

    return fitness

#lancer optimisation meta pour trouver les meilleurs paramètres du GA qui optimise le FCM ou le LVQ

fitness_meta = fitness_ga_factory(base_fitness, taille_adn_problem=10)

meta_ga = AlgorithmeGenetique(
    fonction_fitness=fitness_meta,
    taille_adn=12,
    taille_population=20,
    max_generations=30
)

best_adn, best_score = meta_ga.run()

print("Best GA ADN:", best_adn)
print("Score:", best_score)

print("Params GA:", decode_adn_ga(best_adn))