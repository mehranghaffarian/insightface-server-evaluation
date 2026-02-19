import time
import requests
import json
import os
from tqdm import tqdm

from lfw_loader import load_lfw
from cplfw_loader import load_cplfw
from calfw_loader import load_calfw
from metrics import compute_metrics

SERVER_URL = "http://127.0.0.1:8000/verify"

import json
import os

def save_results(dataset_name, model_name, threshold, accuracy, fmr, fnmr, total_time, count, output_file="results.json"):
    total_time_per_pair = total_time / count

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "FMR": round(fmr, 4),
        "FNMR": round(fnmr, 4),
        "count": count,
        "total_time_sec": round(total_time, 2),
        "avg_time_per_pair_sec": round(total_time_per_pair, 4)
    }

    # if file exists, load old data
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(result)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

def load_dataset(dataset_name, dataset_dir, pairs_file):
    if dataset_name.lower() == "lfw":
        return load_lfw(dataset_dir, pairs_file)

    elif dataset_name.lower() == "cplfw":
        return load_cplfw(dataset_dir, pairs_file)

    elif dataset_name.lower() == "calfw":
        return load_calfw(dataset_dir, pairs_file)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def evaluate(dataset_name, dataset_dir, pairs_file, model_name, threshold=0.4):
    print(f"\nEvaluating {model_name} on {dataset_name}...")
    pairs = load_dataset(dataset_name, dataset_dir, pairs_file)

    print(f"Total pairs loaded: {len(pairs)}")

    similarities = []
    labels = []

    start_time = time.time()

    for img1_path, img2_path, label in tqdm(pairs):

        try:
            with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:

                files = {
                    "image1": f1,
                    "image2": f2
                }

                data = {
                    "model_name": model_name
                }

                response = requests.post(SERVER_URL, files=files, data=data)

                if response.status_code != 200:
                    print(f"Server error for pair: {img1_path}, {img2_path}")
                    continue

                result = response.json()
                similarity = result["similarity"]

                similarities.append(similarity)
                labels.append(label)

        except Exception as e:
            print(f"Error processing pair: {img1_path}, {img2_path}")
            print(str(e))
            continue

    total_time = time.time() - start_time

    accuracy, fmr, fnmr = compute_metrics(similarities, labels, threshold)

    save_results(dataset_name, model_name, threshold, accuracy, fmr, fnmr, total_time, len(pairs))
    print(f"Results saved to results.json")

    return accuracy, fmr, fnmr


if __name__ == "__main__":
    evaluate(
    dataset_name="lfw",
    dataset_dir="datasets/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_file="datasets/lfw/pairs.csv",
    model_name="buffalo_l",
    )
    # evaluate(
    # dataset_name="calfw",
    # dataset_dir="datasets/calfw/images&landmarks/images&landmarks/images",
    # pairs_file="datasets/calfw/pairs_CALFW.txt",
    # model_name="buffalo_l",
    # )
    # evaluate(
    # dataset_name="cplfw",
    # dataset_dir="datasets/cplfw/images",
    # pairs_file="datasets/cplfw/pairs_CPLFW.txt",
    # model_name="buffalo_l",
    # )

    # evaluate(
    # dataset_name="lfw",
    # dataset_dir="datasets/lfw/lfw-deepfunneled/lfw-deepfunneled",
    # pairs_file="datasets/lfw/pairs.csv",
    # model_name="buffalo_s",
    # )
    # evaluate(
    # dataset_name="cplfw",
    # dataset_dir="datasets/cplfw/images",
    # pairs_file="datasets/cplfw/pairs_CPLFW.txt",
    # model_name="buffalo_s",
    # )
    # evaluate(
    # dataset_name="calfw",
    # dataset_dir="datasets/calfw/images&landmarks/images&landmarks/images",
    # pairs_file="datasets/calfw/pairs_CALFW.txt",
    # model_name="buffalo_s",
    # )


