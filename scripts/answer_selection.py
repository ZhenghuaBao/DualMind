import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import load_json
import json

if __name__ == "__main__":
    core_answers = load_json("output/complete_core_results_source.json")
    fallback_answers = load_json("output/context_fallback_results_source.json")

    final_answers = []

    for idx, core_answer in enumerate(core_answers):
        # Date: [ "I can't", "I'm sorry", "I don't", "I'm unable to"]
        # Location: ["Unknown location", ""I'm unable to", ""Not specified."]
        invalid_answers = [ "I can't", "I'm sorry", "I don't", "I'm unable to", "Unknown location", "Not specified."]
        if any(core_answer["output"].startswith(answer) for answer in invalid_answers):
            final_answers.append(fallback_answers[idx])
        else:
            final_answers.append(core_answer)
    
    with open("output/final_results_source.json", "w", encoding="utf-8") as file:
        json.dump(final_answers, file, indent=4)