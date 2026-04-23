import argparse
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run agglomerative clustering over a similarity matrix.")
    parser.add_argument(
        "--similarity-matrix-path",
        default="data/screen_similarity_matrix.npy",
        help="Path to the numpy similarity matrix file.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Threshold values to evaluate.",
    )
    return parser.parse_args()


def load_distance_matrix(similarity_matrix_path):
    print("Loading similarity matrix...")
    sim = np.load(similarity_matrix_path)
    print("Similarity matrix shape:", sim.shape)

    distance_matrix = 1.0 - sim
    distance_matrix[distance_matrix < 0] = 0.0
    np.fill_diagonal(distance_matrix, 0.0)
    print("Distance matrix adjusted for numerical stability.")
    del sim
    return distance_matrix


def create_clusters(distance_matrix, threshold):
    print(f"Creating clusters with threshold {threshold}")
    n = distance_matrix.shape[0]

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
                if rank[root_x] == rank[root_y]:
                    rank[root_y] += 1

    rows, cols = np.where(distance_matrix < threshold)
    for i, j in tqdm(zip(rows, cols), total=len(rows)):
        if i < j:
            union(i, j)

    # First: group by root
    root_to_members = {}
    for i in range(n):
        root = find(i)
        if root not in root_to_members:
            root_to_members[root] = []
        root_to_members[root].append(i)

    # Second: reindex clusters to be consecutive
    clusters = {}
    for new_label, (_, members) in enumerate(root_to_members.items()):
        clusters[new_label] = members

    print("Manual clusters with threshold:", threshold)
    print("Number of clusters formed:", len(clusters))
    # for cluster_id, members in clusters.items():
    #     print(f"Cluster {cluster_id}: {len(members)} screenshots")

    # compute assigned labels
    n = distance_matrix.shape[0]
    labels = np.empty(n, dtype=int)
    for cluster_id, members in clusters.items():
        for idx in members:
            labels[idx] = cluster_id

    # compute silhouette score
    if len(clusters) < 2:
        print(f"Only one cluster formed with threshold {threshold}, silhouette score is not defined.")
        score = -1
    else:
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        print(f"Silhouette Score with threshold {threshold}: {score}")

    with open(f"output/clusters_min_distance_threshold_{threshold}.txt", "w") as f:
        f.write("SILHOUETTE SCORE: " + str(score) + "\n")
        f.write("LABELS: " + str(labels.tolist()) + "\n")
        f.write(f"Number of clusters formed: {len(clusters)}\n")
        for cluster_id, members in clusters.items():
            f.write(f"Cluster {cluster_id}: {len(members)} screenshots\n")


def main():
    args = parse_args()
    distance_matrix = load_distance_matrix(args.similarity_matrix_path)

    os.makedirs("output", exist_ok=True)

    print("Running clustering with different thresholds...")
    Parallel(n_jobs=-1)(
        delayed(create_clusters)(distance_matrix, threshold) for threshold in args.thresholds
    )


if __name__ == "__main__":
    main()
