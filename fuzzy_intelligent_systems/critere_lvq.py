# a minimizer :
def quantization_error(X, model):
    total = 0
    for x in X:
        distances = np.linalg.norm(model.prototypes - x, axis=1)
        total += np.min(distances)
    return total / len(X)

def prototype_stability(X, y, params):
    model1 = LVQ(**params)
    model1.fit(X, y)

    model2 = LVQ(**params)
    model2.fit(X, y)

    return np.linalg.norm(model1.prototypes - model2.prototypes)


# a maximizer :
def accuracy_score(X, y, model):
    preds = model.predict(X)
    return np.mean(preds == y)

def fuzzy_membership(x, prototypes, m=2):
    dist = np.linalg.norm(prototypes - x, axis=1) + 1e-10
    inv = 1.0 / dist
    u = inv / np.sum(inv)
    return u


def fuzzy_silhouette(X, model):
    s_total = 0

    for x in X:
        u = fuzzy_membership(x, model.prototypes)

        a = np.sum(u * np.linalg.norm(model.prototypes - x, axis=1))

        b = np.min([
            np.linalg.norm(x - p)
            for p in model.prototypes
        ])

        s = (b - a) / max(a, b)
        s_total += s

    return s_total / len(X)

