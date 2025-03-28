from dotenv import load_dotenv
import os
from openai import OpenAI
from tqdm import tqdm
import time
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.llm_prompting import (
    gpt4_vision_prompting,
    gpt4_keyword_prompting,
    gpt4_prompting,
)
from utils import *
from dataset_collection.scrape_utils import *
import argparse

import requests


# Function to perform web search
def keyword_search(query, serper_api_key, num_results=10):
    url = "https://google.serper.dev/search/images"

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    payload = {"q": query, "num": num_results}  # Increase the number of results

    response = requests.post(url, json=payload, headers=headers)

    results = []

    if response.status_code == 200:
        data = response.json()
        # Extract images from search results
        for result in data.get("images", []):
            title = result.get("title")
            image = result.get("imageUrl")
            thumbnail = result.get("thumbnailUrl")
            source = result.get("source")
            link = result.get("link")
            position = result.get("position")

            r = {
                "title": title,
                "source": source,
                "link": link,
                "image": image,
                "position": position,
            }

            results.append(r)

        return results
    else:
        print("Error:", response.status_code, response.text)
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect evidence using Google Web Search."
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=" ",  # Provide your own key here as default value
        help="Your key to access the OpenAI services, including the chat API.",
    )
    parser.add_argument(
        "--serper_api_key",
        type=str,
        default=" ",  # Provide your own key here as default value
        help="Your Serper API key to access the Google Web Search services",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="dataset/processed_img/",
        help="The folder where the images are stored.",
    )
    parser.add_argument(
        "--scrape_with_trafilatura",
        type=int,
        default=1,
        help="Whether to scrape the evidence URLs with trafilatura. If 0, it is assumed that a file containing the scraped webpages already exists.",
    )
    parser.add_argument(
        "--trafilatura_path",
        type=str,
        default="dataset/retrieval_results/final_keyword_trafilatura_data.json",
        help="The json file to store the scraped trafilatura content as a json file.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="dataset/retrieval_results/final_keyword_evidence.json",
        help="The json file to store the text keyword evidence as a json file.",
    )
    parser.add_argument(
        "--collect_keyword",
        type=int,
        default=0,
        help="Collect evidence using Google Search.",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=30,
        help="The maximum number of web-pages to collect with the web detection API.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=3,
        help="The waiting time between two web detection API calls",
    )

    # To load api keys or any environmental variable from .env
    load_dotenv()

    args = parser.parse_args()
    serper_api_key = os.getenv("SERPER_API_KEY")

    # Create directories if they do not exist yet
    if not "retrieval_results" in os.listdir("dataset/"):
        os.mkdir("dataset/retrieval_results/")

    if args.collect_keyword:
        raw_keyword_results = []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = "Please describe the context of the given image."

        valid_image_paths = get_valid_images(args.image_path)
        c = 1
        for path in tqdm(valid_image_paths):
            # Generate STORY from the image
            story = gpt4_vision_prompting(prompt, client, path, max_tokens=200)
            # Generate a suitable query for web search
            query_prompt = (
                "Please formulate the given image description into a suitable web search query. /n Description: "
                + str(story)
                + "/n Output: Return reformulated web search query."
            )
            keyword_query, _ = gpt4_prompting(query_prompt, client)

            # Getting keyword search results
            keyword_search_results = keyword_search(keyword_query, serper_api_key)

            for r in keyword_search_results:
                merged_dict = {
                    "image path": path,
                    "story": story,
                    "query": keyword_query,
                    **r,  # merges all key/value pairs from r
                }
                raw_keyword_results.append(merged_dict)
            if c % 5 == 0:
                if os.path.exists(args.json_path):
                    existing_data = load_json(args.json_path)
                    existing_data.extend(raw_keyword_results)
                    with open(args.json_path, "w", encoding="utf-8") as file:
                        # Save raw results
                        json.dump(existing_data, file, indent=4)
                        raw_keyword_results = []
                else:
                    with open(args.json_path, "w", encoding="utf-8") as file:
                        # Save raw results
                        json.dump(raw_keyword_results, file, indent=4)
                        raw_keyword_results = []
            c += 1
            time.sleep(args.sleep)
        existing_data = load_json(args.json_path)
        existing_data.extend(raw_keyword_results)

        with open(args.json_path, "w", encoding="utf-8") as file:
            # Save raw results
            json.dump(existing_data, file, indent=4)

    selected_data = load_json(args.json_path)

    urls = [d["link"] for d in selected_data]
    images = [d["image"] for d in selected_data]
    image_paths = [d["image path"] for d in selected_data]
    story = [d["story"] for d in selected_data]
    query = [d["query"] for d in selected_data]

    if args.scrape_with_trafilatura:
        # Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            result = extract_keyword_info_trafilatura(urls[u], images[u])
            if isinstance(result, str):
                pass
            else:
                output.append(
                    {
                        "image path": image_paths[u],
                        "story": story[u],
                        "query": query[u],
                        **result,
                    }
                )

            # Only store in json file every 50 evidence
            if u % 50 == 0:
                save_keyword_result(output, args.trafilatura_path)
                output = []

    # NOTE: After that we need to download keyword images into keyword_images folder -> download_keyword_search_images.py
