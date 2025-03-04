import argparse
import os
import sys
import json
import numpy as np

from openai import OpenAI
from baseline.answer_generation import run_fallback_model
from baseline.generation_utils import get_topk_evidence
from utils import load_json
from dotenv import load_dotenv


sys.path.insert(1, os.path.join(sys.path[0], ".."))


# Export your OpenAI API in your environment for later use

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 5 pillars answers with LLMs."
    )
    parser.add_argument(
        "--map_manipulated_original",
        type=str,
        default="dataset/map_manipulated_original.json",
        help="Path to the file that maps manipulated images to their identified original version.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="output/keyword_results_source.json",
        help="Path to store the predicted answers.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="source",
        help="The task to perform. One of [source, date, location, motivation]",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="multimodal",
        help="Which input modality to use. One of [vision, evidence, multimodal]",
    )
    parser.add_argument(
        "--n_shots", type=int, default=0, help="How many demonstrations to include."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        help="Which LLM to use for generating answers.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="The maximum number of tokens to generate as output.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="The temperature of the model. Lower values make the output more deterministic.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=5,
        help="The waiting time between two answer generation.",
    )

    load_dotenv()

    args = parser.parse_args()

    if args.model == "gpt4":
        client = OpenAI()
    else:
        client = None

    if "output" not in os.listdir():
        os.mkdir("output/")
    map_manipulated = load_json(args.map_manipulated_original)
    try:
        results_json = load_json(args.results_file)
    except:
        # file does not exist yet
        if not os.path.exists(args.results_file):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(args.results_file, "w") as file:
                json.dump([], file)
        results_json = load_json(args.results_file)
    # Prepare data
    train = load_json("dataset/train.json")
    # Load test images
    test = load_json("dataset/test.json")
    task_test = [t for t in test if t[args.task] != "not enough information"]
    image_paths = [t["image path"] for t in task_test]
    if args.task == "date":
        ground_truth = [t["date numeric label"] for t in task_test]
    else:
        ground_truth = [t[args.task] for t in task_test]

    # Load embeddings and evidence
    clip_keyword_evidence_embeddings = np.load(
        "dataset/embeddings/keyword_evidence_embeddings.npy"
    )
    image_embeddings = np.load("dataset/embeddings/image_embeddings.npy")
    image_embeddings_map = load_json("dataset/embeddings/image_embeddings_map.json")
    keyword_evidence = load_json("dataset/retrieval_results/keyword_evidence.json")
    # Select evidence and demonstrations
    evidence_idx = []
    if args.modality in ["evidence", "multimodal"]:
        for i in range(len(image_paths)):
            evidence_idx.append(
                get_topk_evidence(
                    image_paths[i],
                    keyword_evidence,
                    image_embeddings,
                    clip_keyword_evidence_embeddings,
                    image_embeddings_map,
                )
            )

    # Run the main loop
    run_fallback_model(
        image_paths,
        args.task,
        ground_truth,
        args.results_file,
        map_manipulated,
        args.model,
        keyword_evidence,
        evidence_idx,
        client,
        args.max_tokens,
        args.temperature,
        args.sleep,
    )
