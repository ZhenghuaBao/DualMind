import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import load_json
import json

if __name__ == "__main__":
    keyword_evidence = load_json("dataset/retrieval_results/keyword_evidence.json")

    storys = []
    for p in os.listdir('dataset/test_img'):
        file_path = 'dataset/test_img/' + str(p)
        for story in keyword_evidence:
            if story["image path"] == file_path:
                storys.append({
                    "image path": story["image path"],
                    "story": story['story']
                    })
                break


        
    with open("dataset/retrieval_results/storys.json", "w", encoding="utf-8") as file:
        json.dump(storys, file, indent=4)