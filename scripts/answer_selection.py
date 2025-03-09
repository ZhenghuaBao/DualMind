import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import load_json
import json

if __name__ == "__main__":
    core_answers = load_json("output/original_results_location.json")
    fallback_answers = load_json("output/keyword_results_location.json")

    final_answers = []

    for idx, core_answer in enumerate(core_answers):
        if core_answer["output"] == "NaN":
            final_answers.append(fallback_answers[idx])
        else:
            final_answers.append(core_answer)
    
    with open("output/final_results_location.json", "w", encoding="utf-8") as file:
        json.dump(final_answers, file, indent=4)