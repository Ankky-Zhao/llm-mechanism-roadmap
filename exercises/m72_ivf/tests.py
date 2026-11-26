def evaluate(topics):
    x, queries = make_data(topics=topics)
    index = IVFIndex(kmeans(x, 256)); index.add(x)
    out = {}
    for nprobe in (1, 4, 16, 64, 256):
        recall, compared = 0.0, 0
        for q in queries:
            ids, n = index.search(q, 10, nprobe)
            recall += len(set(ids.tolist()) & set(flat_search(x, q, 10).tolist())) / 10
            compared += n
        out[nprobe] = (recall / len(queries), compared / len(queries) / len(x))
    return x, queries, index, out

x, queries, index, clustered = evaluate(topics=64)
assert sum(len(l) for l in index.lists) == len(x) and len(index.lists) == 256
ids, n = index.search(queries[0], 10, 256)
assert n == len(x) + 256 and set(ids.tolist()) == set(flat_search(x, queries[0], 10).tolist()), "probing every cluster is exact search"
ids, n = index.search(queries[0], 10, 1)
assert n < 0.2 * len(x) and len(ids) <= 10

print("clustered data    nprobe -> (recall@10, share of vectors compared)")
for k, (r, c) in clustered.items(): print(f"  {k:>3}  {r:.3f}  {c:.3f}")
_, _, _, uniform = evaluate(topics=4096)
print("data without cluster structure")
for k, (r, c) in uniform.items(): print(f"  {k:>3}  {r:.3f}  {c:.3f}")

r = [clustered[k][0] for k in (1, 4, 16, 64, 256)]
assert all(a <= b + 1e-9 for a, b in zip(r, r[1:])) and r[-1] == 1.0, "recall grows with nprobe"
assert clustered[4][0] > 0.9 and clustered[4][1] < 0.15, "on clustered data a few probes are enough"
assert uniform[4][0] < 0.6 and uniform[64][0] > 0.8, "without cluster structure far more probes are needed"

try:
    import faiss
    flat = faiss.IndexFlatIP(x.shape[1]); flat.add(x)
    _, I = flat.search(queries[:20], 10)
    assert all(set(I[i].tolist()) == set(flat_search(x, queries[i], 10).tolist()) for i in range(20))
    print("exact search agrees with faiss.IndexFlatIP")
except ImportError:
    print("faiss is not installed, the cross-check was skipped")
print("ok")
