import argparse
from openai import OpenAI
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.answer_generation import *
from baseline.generation_utils import *
from baseline.llm_prompting import *

from dotenv import load_dotenv


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
        default="output/results.json",
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
        default="vision",
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

    args = parser.parse_args()

    load_dotenv()

    if args.model == "gpt4":
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
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
    test = load_json("dataset/test_test.json")
    task_test = [t for t in test if t[args.task] != "not enough information"]
    image_paths = [t["image path"] for t in task_test]
    if args.task == "date":
        ground_truth = [t["date numeric label"] for t in task_test]
    else:
        ground_truth = [t[args.task] for t in task_test]

    # Load embeddings and evidence
    clip_evidence_embeddings = np.load(
        "dataset/embeddings/core_pipeline_evidence_embeddings.npy"
    )
    image_embeddings = np.load("dataset/embeddings/image_embeddings.npy")
    image_embeddings_map = load_json("dataset/embeddings/image_embeddings_map.json")
    evidence = load_json("dataset/retrieval_results/core_pipeline_test_evidence.json")
    # Select evidence and demonstrations
    evidence_idx = []
    if args.modality in ["evidence", "multimodal"]:
        for i in range(len(image_paths)):
            evidence_idxs = get_topk_evidence(
                image_paths[i],
                evidence,
                image_embeddings,
                clip_evidence_embeddings,
                image_embeddings_map,
            )
            evidence_idx.append(evidence_idxs)

    # Select demonstrations
    # Keep train images that have evidence
    images_with_evidence = [ev["image path"] for ev in evidence]
    demonstration_candidates = [
        t for t in train if t["image path"] in images_with_evidence
    ]
    demonstrations = []
    for i in range(len(image_paths)):
        if args.n_shots > 0:

            demonstrations_idx = get_topk_demonstrations(
                image_paths[i],
                args.task,
                demonstration_candidates,
                image_embeddings,
                image_embeddings_map,
                args.n_shots,
            )
            instance_demonstrations = []
            for idx in demonstrations_idx:
                # Get the top k evidence for each demonstration
                demo_image = demonstration_candidates[idx]["image path"]
                demo_answer = demonstration_candidates[idx][args.task]
                demo_evidence_idx = get_topk_evidence(
                    demo_image,
                    evidence,
                    image_embeddings,
                    clip_evidence_embeddings,
                    image_embeddings_map,
                )
                evidence_image_subset = [
                    ev for ev in evidence if ev["image path"] == demo_image
                ]
                demo_evidence = [
                    evidence_image_subset[idx] for idx in demo_evidence_idx
                ]
                instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
            demonstrations.append(instance_demonstrations)
        else:
            demonstrations.append([])

    # Run the main loop
    run_model(
        image_paths,
        args.task,
        ground_truth,
        args.results_file,
        map_manipulated,
        args.modality,
        args.model,
        evidence,
        evidence_idx,
        demonstrations,
        client,
        args.max_tokens,
        args.temperature,
        args.sleep,
    )
