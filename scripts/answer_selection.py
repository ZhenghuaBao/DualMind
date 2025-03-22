import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import load_json
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer selection process.")
    parser.add_argument(
        "--core_answer_path",
        type=str,
        default="output/core_results_date_4o.json",
        help="Path to the output of core answers.",
    )
    parser.add_argument(
        "--fallback_answer_path",
        type=str,
        default="output/final_fallback_results_date.json",
        help="Path to the output of fallback answers.",
    )

    args = parser.parse_args()

    core_answer_path = args.core_answer_path
    fallback_answer_path = args.fallback_answer_path

    core_answers = load_json(core_answer_path)
    fallback_answers = load_json(fallback_answer_path)

    final_answers = []

    for idx, core_answer in enumerate(core_answers):
        # Date: [ "I can't", "I'm sorry", "I don't", "I'm unable to"]
        # Location: ["Unknown location", ""I'm unable to", ""Not specified."]
        invalid_answers = [
            "I can't",
            "I'm sorry",
            "I don't",
            "I'm unable to",
            "The image does not",
            "The photo does not",
            "The image provided does not",
        ]
        if any(core_answer["output"].startswith(answer) for answer in invalid_answers):
            if not any(
                fallback_answers[idx]["output"].startswith(answer)
                for answer in invalid_answers
            ):
                final_answers.append(fallback_answers[idx])
            else:
                final_answers.append(core_answer)
        else:
            final_answers.append(core_answer)

    with open(
        "output/final_results_4omini/final_complete_results_date.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_answers, file, indent=4)
