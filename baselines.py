# baselines.py
import numpy as np
from math import hypot

# GCA: Greedy Competitiveness Allocation (one-shot)
def gca_allocate(task, users, edges, cfg):
    user = users[task.user_id]
    upos = (user.x, user.y)
    candidates = []
    for e in edges:
        d = hypot(upos[0]-e.x, upos[1]-e.y)
        if d <= cfg["coverage_radius"]:
            score = (1.0 / (d + 1e-6)) / (e.price + 1e-6)
            candidates.append((e, d, score))
    if not candidates:
        return ("none", None, None)
    candidates.sort(key=lambda x: x[2], reverse=True)
    chosen = candidates[0][0]
    if chosen.resource_free >= task.cpu_need:
        chosen.resource_free -= task.cpu_need
        chosen.history.append((task.arrival_t, task.cpu_need, task.size_mb))
        return ("edge", chosen.eid, float(task.cpu_need))
    return ("none", None, None)

# MVAC: Multi-round Vickrey Auction without prediction
def mvac_allocate(task, users, edges, cfg, cloud):
    user = users[task.user_id]
    upos = (user.x, user.y)
    candidates = []
    for e in edges:
        d = hypot(upos[0]-e.x, upos[1]-e.y)
        if d <= cfg["coverage_radius"]:
            candidates.append((e, d))
    if not candidates:
        if cloud.resource_free >= task.cpu_need:
            cloud.resource_free -= task.cpu_need
            return ("cloud", None, float(task.cpu_need))
        return ("none", None, None)
    bids = [(e, e.price) for e, _ in candidates]
    bids_sorted = sorted(bids, key=lambda x: x[1], reverse=True)
    winner = bids_sorted[0][0]
    if cloud.resource_free >= task.cpu_need:
        cloud.resource_free -= task.cpu_need
        winner.resource_free += task.cpu_need
        if winner.resource_free >= task.cpu_need:
            winner.resource_free -= task.cpu_need
            winner.history.append((task.arrival_t, task.cpu_need, task.size_mb))
            return ("edge", winner.eid, float(task.cpu_need))
    if cloud.resource_free >= task.cpu_need:
        cloud.resource_free -= task.cpu_need
        return ("cloud", None, float(task.cpu_need))
    return ("none", None, None)

# DPKA: Distance+Price K-means Approx (uses sklearn)
def dpka_allocate(task, users, edges, cfg, k=3):
    try:
        from sklearn.cluster import KMeans
    except Exception:
        # fallback to simple greedy if sklearn missing
        return gca_allocate(task, users, edges, cfg)
    user = users[task.user_id]
    upos = (user.x, user.y)
    feats = []
    idxs = []
    for e in edges:
        d = hypot(upos[0]-e.x, upos[1]-e.y)
        if d <= cfg["coverage_radius"]:
            feats.append([d, e.price])
            idxs.append(e)
    if not feats:
        return ("none", None, None)
    X = np.array(feats)
    kk = min(k, len(X))
    kmeans = KMeans(n_clusters=kk, random_state=42).fit(X)
    labels = kmeans.labels_
    cluster_scores = []
    for c in range(kk):
        members = X[labels==c]
        cluster_scores.append(members.mean())
    best_c = int(np.argmin([s.sum() for s in cluster_scores]))
    candidates = [idxs[i] for i,l in enumerate(labels) if l==best_c]
    candidates.sort(key=lambda e: e.price)
    chosen = candidates[0]
    if chosen.resource_free >= task.cpu_need:
        chosen.resource_free -= task.cpu_need
        chosen.history.append((task.arrival_t, task.cpu_need, task.size_mb))
        return ("edge", chosen.eid, float(task.cpu_need))
    return ("none", None, None)
