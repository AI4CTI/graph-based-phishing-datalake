# WP2_FISA Deliverable: Screenshot Encoding and Clustering

This project builds visual embeddings from website screenshots and clusters them using two agglomerative-style strategies:

- Min-distance merging over a precomputed similarity matrix
- Centroid-based iterative merging directly on embeddings

The pipeline is designed for screenshots stored in parquet files (column `screenshots`), where each screenshot payload is expected to contain base64-encoded image data.

## Project Structure

```text
WP2_FISA/
├── screen_encoding.py
├── agglomerative_clustering_min_distance.py
├── agglomerative_clustering_centroids.py
├── utils.py
├── data/
│   ├── .gitignore
│   ├── screen_similarity_matrix.npy          # generated
│   └── valid_screenshots.txt                # generated
├── output/
│   ├── .gitignore
│   └── clusters_*.txt                       # generated
└── README.md
```

## Requirements

- Python 3.9+
- Optional GPU for faster embedding inference

Install dependencies:

```bash
pip install pandas pillow torch transformers tqdm numpy scikit-learn joblib pyarrow
```

## Input Data Expectations

The input directory (default: `data/sample/history__web_crawler/`) should contain parquet files with at least:

- `domain`
- `screenshots`

`screenshots` is expected to be iterable-like per row, with the first element containing a base64 image string (used as `screenshot[0]`).

## Scripts

### 1) Generate Embeddings + Similarity Matrix

Script: `screen_encoding.py`

What it does:

- Loads parquet files and reads screenshots
- Encodes images using a Hugging Face vision model
- Computes pairwise cosine similarity between screenshot embeddings
- Saves:
	- similarity matrix (`.npy`)
	- optional raw embeddings (`.npy`)
	- list of valid screenshots (`.txt`)

Run:

```bash
python screen_encoding.py \
	--input-dir data/sample/history__web_crawler/ \
	--vision-encoder-name google/vit-base-patch16-224-in21k \
	--cache-dir /data2/hf_cache/models \
	--output-npy data/screen_similarity_matrix.npy \
	--save-embeddings
```

Useful options:

- `--max-files N`: process only the first `N` parquet files

Arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | `data/sample/history__web_crawler/` | Directory containing parquet files with the `screenshots` column. |
| `--vision-encoder-name` | `str` | `google/vit-base-patch16-224-in21k` | Hugging Face vision model used to encode screenshots. |
| `--cache-dir` | `str` | `/data2/hf_cache/models` | Local Hugging Face cache directory for model files. |
| `--output-npy` | `str` | `data/screen_similarity_matrix.npy` | Output path for the similarity matrix in `.npy` format. |
| `--max-files` | `int` | `None` | Optional limit on how many parquet files are loaded. |
| `--save-embeddings` | `flag` | `False` | If set, also saves screenshot embeddings to `data/screen_embeddings.npy`. |

### 2) Agglomerative Clustering (Min Distance)

Script: `agglomerative_clustering_min_distance.py`

What it does:

- Loads a precomputed similarity matrix
- Converts it to distance (`1 - similarity`)
- Builds clusters using a union-find strategy for each threshold
- Computes silhouette score (precomputed distance metric)
- Writes one output report per threshold

Run:

```bash
python agglomerative_clustering_min_distance.py \
	--similarity-matrix-path data/screen_similarity_matrix.npy \
	--thresholds 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5
```

Output files are written to `output/` with names like:

- `clusters_min_distance_threshold_<threshold>.txt`

Arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--similarity-matrix-path` | `str` | `data/screen_similarity_matrix.npy` | Path to the precomputed similarity matrix (`.npy`). |
| `--thresholds` | `list[float]` | `0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5` | One or more merge thresholds to evaluate. |

### 3) Alternative Method: Agglomerative Clustering (Centroid-Based)

Script: `agglomerative_clustering_centroids.py`

What it does:

- Uses embeddings (either precomputed or generated on the fly)
- Iteratively merges the closest pair of active cluster centroids
- Stops when minimum centroid distance exceeds threshold
- Computes silhouette score and writes per-threshold reports

This method is an alternative to the min-distance approach above, not a required next step.

Run with precomputed embeddings:

```bash
python agglomerative_clustering_centroids.py \
	--precomputed-embeddings-path data/screen_embeddings.npy \
	--thresholds 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5
```

Run with on-the-fly embedding generation:

```bash
python agglomerative_clustering_centroids.py \
	--input-dir data/sample/history__web_crawler/ \
	--model-name google/vit-base-patch16-224-in21k \
	--cache-dir /data2/hf_cache/models \
	--thresholds 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5
```

Output files are written to `output/` with names like:

- `clusters_centroids_threshold_<threshold>.txt`

Arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input-dir` | `str` | `data/sample/history__web_crawler/` | Input parquet directory (used when embeddings are not precomputed). |
| `--precomputed-embeddings-path` | `str` | `None` | Optional path to embeddings (`.npy`); skips embedding generation if provided. |
| `--thresholds` | `list[float]` | `0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5` | One or more centroid-distance thresholds to evaluate. |
| `--model-name` | `str` | `google/vit-base-patch16-224-in21k` | Hugging Face model used when generating embeddings on the fly. |
| `--cache-dir` | `str` | `/data2/hf_cache/models` | Local Hugging Face cache directory for model files. |
| `--max-files` | `int` | `None` | Optional limit on how many parquet files are loaded for embedding generation. |

## Typical End-to-End Workflow

1. Generate similarity matrix (and optionally embeddings):

```bash
python screen_encoding.py --input-dir <parquet_dir> --save-embeddings
```

2. Choose one clustering strategy:

Option A: run min-distance clustering:

```bash
python agglomerative_clustering_min_distance.py
```

Option B: run centroid-based clustering:

```bash
python agglomerative_clustering_centroids.py --precomputed-embeddings-path data/screen_embeddings.npy
```

3. Compare silhouette scores in `output/*.txt` across thresholds and methods.

## Notes

- `data/` and `output/` artifacts are intentionally ignored by git (except `.gitignore` placeholders).
- Very large datasets may require significant RAM due to pairwise similarity matrix computation.
- If no valid screenshots are decoded, `screen_encoding.py` raises a runtime error.
