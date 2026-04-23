import pandas as pd
import base64
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from utils import generate_embeddings, read_data


def parse_args():
    parser = argparse.ArgumentParser(description="Compute screenshot similarity with a vision encoder")
    parser.add_argument(
        "--input-dir",
        default="data/sample/history__web_crawler/",
        help="Directory containing parquet files with a 'screenshots' column",
    )
    parser.add_argument(
        "--vision-encoder-name",
        default="google/vit-base-patch16-224-in21k",
        help="Hugging Face model name/path for the vision encoder",
    )
    parser.add_argument(
        "--cache-dir",
        default="/data2/hf_cache/models",
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--output-npy",
        default="data/screen_similarity_matrix.npy",
        help="Output path for similarity matrix in .npy format",
    ) 
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of parquet files to read",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Whether to save the raw embeddings as well (in .npy format)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_web_crawler_df = read_data(args.input_dir, max_files=args.max_files)

    model = AutoModel.from_pretrained(args.vision_encoder_name, cache_dir=args.cache_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.vision_encoder_name, cache_dir=args.cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    screenshots = all_web_crawler_df["screenshots"].tolist()
    print(f"Number of screenshots: {len(screenshots)}")

    screenshots = [s for s in screenshots if s and len(s) > 0]
    print(f"Number of non-empty screenshots: {len(screenshots)}")

    embeddings, valid_screenshots = generate_embeddings(screenshots, model, image_processor, device)

    if len(embeddings) == 0:
        raise RuntimeError("No valid embeddings were produced from input screenshots")

    # save valid screenshots for reference
    with open("data/valid_screenshots.txt", "w") as f:
        for s in valid_screenshots:
            f.write(s[0] + "\n")

    embeddings = np.array(embeddings)
    sim = cosine_similarity(embeddings).astype(np.float16)
    np.save(args.output_npy, sim)

    if args.save_embeddings:
        np.save("data/screen_embeddings.npy", embeddings)


if __name__ == "__main__":
    main()
