import sys
import os
import json
import shutil
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import download_images, load_json

# NOTE: This file is needed to download images for collected results of keyword search

if __name__ == "__main__":
    directory = "dataset/final_keyword_images"

    # Clear the folder before downloading new images. Uncomment if needed
    # if os.path.exists(directory):
    #     shutil.rmtree(directory)  # Remove the entire folder
    # os.makedirs(directory)  # Recreate the empty folder

    keyword_evidence = load_json(
        "dataset/retrieval_results/final_keyword_trafilatura_data.json"
    )

    images = [ev["image"] for ev in keyword_evidence]
    counter = 1

    for image in tqdm(images):
        current_img = str(counter) + ".jpg"
        img_path = os.path.join(directory, current_img)
        if os.path.exists(img_path):
            counter += 1
            continue
        download_images(image, str(counter) + ".jpg", directory)
        counter += 1

    directory = "dataset/final_keyword_images"

    for idx, evidence in enumerate(keyword_evidence):
        file_name = f"{idx + 1}.jpg"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            evidence["downloaded"] = True
        else:
            evidence["downloaded"] = False

    with open(
        "dataset/retrieval_results/final_processed_keyword_trafilatura_data.json",
        "w",
        encoding="utf-8",
    ) as file:
        # Save raw results
        json.dump(keyword_evidence, file, indent=4)
