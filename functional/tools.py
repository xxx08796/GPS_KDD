import numpy as np


def calculate_embed(graph):
    embed_scores = np.empty(len(graph.nodes()))
    for u in graph.nodes():
        neighbors = set(graph.neighbors(u))
        if len(neighbors) == 0:
            embed_scores[u] = 0
            continue
        embed_sum = 0
        for v in neighbors:
            common_neighbors = neighbors.intersection(set(graph.neighbors(v)))
            union_neighbors = neighbors.union(set(graph.neighbors(v)))
            embed_sum += len(common_neighbors) / len(union_neighbors)
        embed_score = embed_sum / len(neighbors)
        embed_scores[u] = embed_score
    return embed_scores
