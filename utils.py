from tqdm import tqdm
from PIL import Image
import base64
import torch
import io
import os
import pandas as pd


def generate_embeddings(screenshots, model, image_processor, device):
    embeddings = []
    valid_screenshots = []
    for screenshot in tqdm(screenshots):
        try:
            img_bytes = base64.b64decode(screenshot[0])
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
        except Exception:
            print(f"Error decoding image: {screenshot}")
            continue

        inputs = image_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
        embeddings.append(cls)
        valid_screenshots.append(screenshot)
    return embeddings, valid_screenshots


def read_data(input_dir, max_files=None):
    web_crawler_files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    if max_files is not None:
        web_crawler_files = web_crawler_files[:max_files]
    print("Total web crawler files:", len(web_crawler_files))

    all_web_crawler_df = pd.DataFrame()
    for file in tqdm(web_crawler_files):
        df = pd.read_parquet(os.path.join(input_dir, file))
        if all_web_crawler_df.empty:
            all_web_crawler_df = df[["domain", "screenshots"]]
        else:
            all_web_crawler_df = pd.concat([all_web_crawler_df, df[["domain", "screenshots"]]], ignore_index=True)

    print("Total rows in combined dataframe:", len(all_web_crawler_df))
    return all_web_crawler_df