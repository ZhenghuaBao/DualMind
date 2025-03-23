import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from utils import load_json
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select either core or fallback answer."
    )

    parser.add_argument(
        "--task",
        type=str,
        default="source",
        help="The task to perform. One of [source, date, location, motivation]",
    )

    args = parser.parse_args()

    task = args.task

    core_path = f"output/final_results_tineye/results_{task}_gpt4.json"
    fallback_path = (
        f"output/final_results_tineye/final_fallback_results_turbo_{task}.json"
    )

    core_answers = load_json(core_path)
    fallback_answers = load_json(fallback_path)

    final_answers = []
    replaced_img = []

    for idx, core_answer in enumerate(core_answers):
        # Date: [ "I can't", "I'm sorry", "I don't", "I'm unable to"]
        # Location: ["Unknown", ""I'm unable to", ""Not specified."]
        # LLaVA:  invalid_answers = [ "True", "I'm sorry", "Unrelated", "No"]
        invalid_answers = [
            "not" "I can't",
            "I'm sorry",
            "I don't",
            "I'm unable to",
            "The image does not",
            "The photo does not",
            "The image provided does not",
            "Unknown",
            "Not specified",
        ]
        if any(
            answer.lower() in core_answer["output"].strip().lower()
            for answer in invalid_answers
        ):
            if not any(
                answer.lower() in fallback_answers[idx]["output"].strip().lower()
                for answer in invalid_answers
            ):
                final_answers.append(fallback_answers[idx])
                replaced_img.append(core_answer["img_path"])
                print("Core: " + core_answer["output"])
                print("Fallback: " + fallback_answers[idx]["output"])
            else:
                final_answers.append(core_answer)
        else:
            final_answers.append(core_answer)

    with open(
        f"output/final_results_tineye/final_complete_results_{task}.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_answers, file, indent=4)
