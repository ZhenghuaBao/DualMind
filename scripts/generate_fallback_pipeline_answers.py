import argparse
import os
import sys
import json
import numpy as np

from openai import OpenAI

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.answer_generation import run_fallback_model
from baseline.generation_utils import (
    get_topk_keyword_evidence,
    cosine_similarity,
    get_topk_story,
)
from utils import load_json, classify_similarity, classify_confidence_level
from dotenv import load_dotenv
from urllib.parse import urlparse


sys.path.insert(1, os.path.join(sys.path[0], ".."))


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
    test = load_json("dataset/test.json")
    task_test = [t for t in test if t[args.task] != "not enough information"]
    image_paths = [t["image path"] for t in task_test]
    if args.task == "date":
        ground_truth = [t["date numeric label"] for t in task_test]
    else:
        ground_truth = [t[args.task] for t in task_test]

    # Load embeddings and evidence
    clip_keyword_evidence_embeddings = np.load(
        "dataset/embeddings/content_keyword_evidence_embeddings.npy"
    )
    image_embeddings = np.load("dataset/embeddings/image_embeddings.npy")
    image_embeddings_map = load_json("dataset/embeddings/image_embeddings_map.json")
    keyword_evidence = load_json(
        "dataset/retrieval_results/final_processed_keyword_trafilatura_data.json"
    )
    clip_keyword_images_embeddings = np.load(
        "dataset/embeddings/final_keyword_image_embeddings.npy"
    )
    clip_story_embeddings = np.load("dataset/embeddings/final_story_embeddings.npy")
    keyword_image_embeddings_map = load_json(
        "dataset/embeddings/final_keyword_image_embeddings_map.json"
    )

    # This is needed for getting the right index of the image
    keyword_images = [
        f for f in os.listdir("dataset/final_keyword_images") if f.endswith(".jpg")
    ]
    # Select evidence and demonstrations
    evidence_idx = []
    keyword_image_embedding_idices = []
    similarity_level = []
    story_evidence = {}
    confidence_levels = []

    # Mapping dictionary including 'Unknown'
    confidence_map = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 1, "N/A": 1}

    # Loading the MBFC database
    with open("dataset/MBFC Bias Database 12-12-24.json", "r") as file:
        mbfc_data = json.load(file)

    # Create {domain: credibility} quick lookup table
    credibility_lookup = {entry["Domain"]: entry["Credibility"] for entry in mbfc_data}

    def get_credibility(url):
        """Obtain the main domain name from the URL and query the trustworthiness of MBFC."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")  # remove 'www.'
        return credibility_lookup.get(domain, "Unknown")  # if cant find, return Unknown

    for ev in keyword_evidence:
        url = ev["url"]
        confidence = get_credibility(url)
        ev["confidence"] = confidence

    if args.modality in ["evidence", "multimodal"]:
        for i in range(len(image_paths)):
            topk_keyword_evidence = get_topk_keyword_evidence(
                image_paths[i],
                keyword_evidence,
                image_embeddings,
                clip_keyword_evidence_embeddings,
                image_embeddings_map,
            )
            if len(topk_keyword_evidence) == 0:
                evidence_idx.append([])
                # No evidence provided only rely on VLM
                confidence_levels.append("Low")
                continue

            assert (
                len(topk_keyword_evidence) > 0
            ), f"Did not found any matching evidence for image {image_paths[i]}."

            evidence = [
                ev
                for ev in keyword_evidence
                if ev["image path"] == image_paths[i] and ev["downloaded"]
            ]
            confidence_list = []
            for ev_idx in topk_keyword_evidence:
                confidence_list.append(evidence[ev_idx]["confidence"])

            # Convert list while ignoring 'Unknown'
            confidence_score = np.mean(
                [confidence_map[value] for value in confidence_list]
            )
            confidence_level = classify_confidence_level(confidence_score)

            keyword_image_embedding_idx_local = []
            for ev_idx in topk_keyword_evidence:
                keyword_image_embedding_idx = keyword_evidence.index(evidence[ev_idx])
                keyword_image_embedding_idx_local.append(keyword_image_embedding_idx)
            keyword_image_embedding_idices.append(keyword_image_embedding_idx_local)

            score = []
            for idx in keyword_image_embedding_idx_local:
                actual_path = "dataset/final_keyword_images/" + str(idx + 1) + ".jpg"
                actual_idx = int(keyword_image_embeddings_map[actual_path])

                s = cosine_similarity(
                    image_embeddings[i], clip_keyword_images_embeddings[actual_idx]
                )
                score.append(s)
            assert len(score) > 0

            keyword_similarity = classify_similarity(np.mean(score))
            similarity_level.append(keyword_similarity)

            # If the similarity score is low
            # We rely on the STORY and choose top k based on STORY
            if keyword_similarity == "Low":
                story_evidence_idx = get_topk_story(
                    image_paths[i],
                    keyword_evidence,
                    clip_keyword_evidence_embeddings,
                    clip_story_embeddings,
                )
                confidence_list = []
                for ev_idx in story_evidence_idx:
                    confidence_list.append(evidence[ev_idx]["confidence"])
                # Convert list while ignoring 'Unknown'
                confidence_score = np.mean(
                    [confidence_map[value] for value in confidence_list]
                )
                confidence_level = classify_confidence_level(confidence_score)
                story_evidence[i] = story_evidence_idx

            evidence_idx.append(topk_keyword_evidence)

            confidence_levels.append(confidence_level)

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
        story_evidence,
        confidence_levels,
        client,
        args.max_tokens,
        args.temperature,
        args.sleep,
    )
