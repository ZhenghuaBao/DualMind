import sys
import os
import json
import shutil
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import download_images, load_json

# NOTE: Clean keyword_images folder before download

if __name__ == "__main__":
    directory = "dataset/keyword_images"

    # Clear the folder before downloading new images
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the entire folder
    os.makedirs(directory)  # Recreate the empty folder

    keyword_evidence = load_json("dataset/retrieval_results/trafilatura_data_keyword.json")

    images = [ev['image'] for ev in keyword_evidence]
    counter = 1

    for image in images:
        download_images(image, str(counter) + ".jpg")
        counter += 1
    
    directory = "dataset/keyword_images"

    for idx, evidence in enumerate(keyword_evidence):
        file_name = f"{idx + 1}.jpg"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):  
            evidence["downloaded"] = True
        else:
            evidence["downloaded"] = False

    with open("dataset/retrieval_results/processed_trafilatura_data_keyword.json", "w", encoding="utf-8") as file:
            # Save raw results
            json.dump(keyword_evidence, file, indent=4)