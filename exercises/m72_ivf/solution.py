import numpy as np

def make_data(n=4096, d=32, topics=64, n_queries=200, seed=0):
    """Unit vectors scattered around `topics` centres. topics == n gives data without cluster structure."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(topics, d))
    x = centres[rng.integers(topics, size=n)] + 0.7 * rng.normal(size=(n, d))
    q = centres[rng.integers(topics, size=n_queries)] + 0.7 * rng.normal(size=(n_queries, d))
    unit = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)
    return unit(x), unit(q)

def kmeans(x, k, iters=10, seed=0):
    """Spherical k-means: assign by largest inner product, renormalize the centroids."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ c.T, axis=1)
        for j in range(k):
            members = x[assign == j]
            if len(members):
                c[j] = members.mean(0)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c

def flat_search(x, q, k):
    """Exact search: compare the query with every vector. Returns the ids of the k largest inner products."""
    return np.argsort(-(x @ q))[:k]


class IVFIndex:
    def __init__(self, centroids):
        self.centroids = centroids          # (nlist, d)
        self.lists = None                   # one array of vector ids per cluster
        self.x = None

    def add(self, x):
        """Store the vectors and the ids that fall into each cluster."""
        self.x = x
        assign = np.argmax(x @ self.centroids.T, axis=1)
        self.lists = [np.where(assign == j)[0] for j in range(len(self.centroids))]

    def search(self, q, k, nprobe):
        """Return (ids of the k best vectors among the nprobe closest clusters, number of vectors compared)."""
        cells = np.argsort(-(self.centroids @ q))[:nprobe]
        ids = np.concatenate([self.lists[j] for j in cells])
        best = ids[np.argsort(-(self.x[ids] @ q))[:k]]
        return best, len(self.centroids) + len(ids)
