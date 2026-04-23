import torch
from transformers import ViTModel, AutoImageProcessor
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from utils import generate_embeddings, read_data
import argparse
from sklearn.metrics import silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run centroid-based clustering on web crawler screenshots.")
    parser.add_argument(
        "--input-dir",
        default="data/sample/history__web_crawler/",
        help="Directory containing the parquet files to load.",
    )
    parser.add_argument(
        "--precomputed-embeddings-path",
        default=None,
        help="Optional path to precomputed embeddings in .npy format. If provided, will skip embedding generation.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Threshold values to evaluate.",
    )
    parser.add_argument(
        "--model-name",
        default="google/vit-base-patch16-224-in21k",
        help="Hugging Face model name to load.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/data2/hf_cache/models",
        help="Directory used for Hugging Face model caching.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of parquet files to read from the input directory.",
    )
    return parser.parse_args()


def centroid_based_clustering(embeddings, threshold):

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = embeddings / np.clip(norms, 1e-12, None)
    n = X.shape[0]

    embeddings = np.array(embeddings)
    sim = cosine_similarity(embeddings).astype(np.float16)
    distance_matrix = 1.0 - sim
    distance_matrix[distance_matrix < 0] = 0.0
    np.fill_diagonal(distance_matrix, 0.0)

    # Cluster membership
    clusters = {i: [i] for i in range(n)}

    # Centroids as matrix instead of dict
    centroids = X.copy()

    # Active mask
    active = np.ones(n, dtype=bool)

    max_merges = n - 1
    with tqdm(total=max_merges, desc="Merging clusters") as pbar:
        while True:
            active_idx = np.flatnonzero(active)
            if len(active_idx) < 2:
                print("Less than 2 active clusters remaining, stopping.")
                break

            C = centroids[active_idx]  # shape: (k, d)

            # Cosine similarity of normalized centroids
            sim_matrix = C @ C.T
            dist_matrix = 1.0 - sim_matrix

            # Ignore self-pairs
            np.fill_diagonal(dist_matrix, np.inf)

            # Find closest pair
            flat_idx = np.argmin(dist_matrix)
            min_distance = dist_matrix.flat[flat_idx]

            if min_distance >= threshold:
                print(f"Minimum distance {min_distance:.4f} exceeds threshold {threshold}, stopping.")
                break

            i_local, j_local = np.unravel_index(flat_idx, dist_matrix.shape)
            c1 = active_idx[i_local]
            c2 = active_idx[j_local]

            # Merge c2 into c1
            clusters[c1].extend(clusters[c2])

            # Update centroid
            new_centroid = X[clusters[c1]].mean(axis=0)
            new_centroid /= np.clip(np.linalg.norm(new_centroid), 1e-12, None)
            centroids[c1] = new_centroid

            # Remove merged cluster
            del clusters[c2]
            active[c2] = False

            pbar.update(1)

    # Reindex clusters
    final_clusters = {
        new_id: members
        for new_id, members in enumerate(clusters.values())
    }

    print(f"Centroid-based clustering with threshold {threshold}")
    print("Number of clusters formed:", len(final_clusters))
    # for cluster_id, members in final_clusters.items():
    #     print(f"Cluster {cluster_id}: {len(members)} screenshots")

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

    with open(f"output/clusters_centroids_threshold_{threshold}.txt", "w") as f:
        f.write("SILHOUETTE SCORE: " + str(score) + "\n")
        f.write("LABELS: " + str(labels.tolist()) + "\n")
        f.write(f"Number of clusters formed: {len(clusters)}\n")
        for cluster_id, members in clusters.items():
            f.write(f"Cluster {cluster_id}: {len(members)} screenshots\n")



def main():
    args = parse_args()

    if args.precomputed_embeddings_path is not None:
        print(f"Loading precomputed embeddings from {args.precomputed_embeddings_path}...")
        embeddings = np.load(args.precomputed_embeddings_path)
    else:
        print("Loading web crawler data...")
        web_crawler_files = os.listdir(args.input_dir)
        web_crawler_files = [f for f in web_crawler_files if f.endswith(".parquet")]
        all_web_crawler_df = read_data(args.input_dir, max_files=args.max_files)

        model = ViTModel.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        image_processor = AutoImageProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        screenshots = all_web_crawler_df["screenshots"].tolist()
        print(f"Number of screenshots: {len(screenshots)}")

        screenshots = [s for s in screenshots if s and len(s) > 0]
        print(f"Number of non-empty screenshots: {len(screenshots)}")

        embeddings, _ = generate_embeddings(screenshots, model, image_processor, device)
        embeddings = np.array(embeddings)


    threshold_list = parse_args().thresholds   
    _ = Parallel(n_jobs=-1)(
        delayed(centroid_based_clustering)(embeddings, thr) for thr in threshold_list
    )

if __name__ == "__main__":
    main()