import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import load_json
import json

if __name__ == "__main__":
    core_answers = load_json("output/core_results_date_4o.json")
    fallback_answers = load_json("output/final_fallback_results_date.json")

    final_answers = []

    for idx, core_answer in enumerate(core_answers):
        # Date: [ "I can't", "I'm sorry", "I don't", "I'm unable to"]
        # Location: ["Unknown location", ""I'm unable to", ""Not specified."]
        invalid_answers = [ "I can't", "I'm sorry", "I don't", "I'm unable to", "The image does not", "The photo does not", "The image provided does not"]
        if any(core_answer["output"].startswith(answer) for answer in invalid_answers):
            if not any(fallback_answers[idx]["output"].startswith(answer) for answer in invalid_answers):
                final_answers.append(fallback_answers[idx])
            else:
                final_answers.append(core_answer)
        else:
            final_answers.append(core_answer)
    
    with open("output/final_results_4omini/final_complete_results_date.json", "w", encoding="utf-8") as file:
        json.dump(final_answers, file, indent=4)