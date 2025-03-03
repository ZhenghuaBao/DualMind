from dotenv import load_dotenv
from google.cloud import vision
import os
from openai import OpenAI
from tqdm import tqdm
import time
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.llm_prompting import gpt4_vision_prompting, gpt4_prompting
from utils import *
from dataset_collection.scrape_utils import *
import argparse

import requests
import imghdr


def google_search(query, google_api_key, google_cse_id, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": google_api_key,
        "cx": google_cse_id,
        "num": num_results,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()
        search_results = results.get("items", [])

        # Extracting title, link, and image (if available)
        extracted_results = []
        for item in search_results:
            title = item.get("title")
            link = item.get("link")
            image_links = []  # ✅ Initialize as an empty list

            # Check if images exist in pagemap
            if "pagemap" in item:
                pagemap = item["pagemap"]

                # ✅ Extract all image links (if available)
                if "cse_image" in pagemap:
                    image_links = [
                        img["src"] for img in pagemap["cse_image"] if "src" in img
                    ]

            extracted_results.append(
                {
                    "title": title,
                    "link": link,
                    "image": image_links,
                }  # ✅ Now safe to use
            )

        return extracted_results

    else:
        print(f"Error {response.status_code}: {response.text}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect evidence using Google Reverse Image Search."
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=" ",  # Provide your own key here as default value
        help="Your key to access the OpenAI services, including the chat API.",
    )
    parser.add_argument(
        "--google_search_api_key",
        type=str,
        default=" ",  # Provide your own key here as default value
        help="Your key to access the Google Web Search services",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="dataset/test_img/",
        help="The folder where the images are stored.",
    )
    parser.add_argument(
        "--raw_keyword_urls_path",
        type=str,
        default="dataset/retrieval_results/keyword_results.json",
        help="The json file to store the raw keyword search results.",
    )
    parser.add_argument(
        "--evidence_urls",
        type=str,
        default="dataset/retrieval_results/evidence_urls.json",
        help="Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.",
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
        default="dataset/retrieval_results/trafilatura_data_keyword.json",
        help="The json file to store the scraped trafilatura content as a json file.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="dataset/retrieval_results/keyword_evidence.json",
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
        default=10,
        help="The maximum number of web-pages to collect with the web detection API.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=3,
        help="The waiting time between two web detection API calls",
    )

    load_dotenv()

    args = parser.parse_args()
    googel_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    # openai_key = os.getenv("OPENAI_API_KEY")

    # Create directories if they do not exist yet
    if not "retrieval_results" in os.listdir("dataset/"):
        os.mkdir("dataset/keyword_retrieval_results/")

    if args.collect_keyword:
        raw_keyword_results = []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = "Please tell me the story of the given image."

        valid_image_paths = get_valid_images(args.image_path)

        for path in tqdm(valid_image_paths):
            # Generate STORY from the image
            story = gpt4_vision_prompting(prompt, client, path)
            # Generate a suitable query for web search
            query_prompt = (
                "Please formulate the given story into a suitable query for web search. /n Story: "
                + story
            )
            keyword_query = gpt4_prompting(query_prompt, client)
            # Getting keyword search results
            keyword_search_results = google_search(
                keyword_query, googel_api_key, google_cse_id
            )

            titles = [r["title"] for r in keyword_search_results]
            urls = [r["link"] for r in keyword_search_results]
            image_urls = [r["image"] for r in keyword_search_results]

            raw_keyword_results.append(
                {
                    "image path": args.image_path + path,
                    "story": story,
                    "titles": titles,
                    "urls": urls,
                    "image urls": image_urls,
                }
            )
            time.sleep(args.sleep)
        with open(args.raw_keyword_urls_path, "w") as file:
            # Save raw results
            json.dump(raw_keyword_results, file, indent=4)
        # Apply filtering to the URLs to remove content produced by FC organizations and content that is not scrapable
        selected_data = get_filtered_retrieval_keyword_results(
            args.raw_keyword_urls_path
        )

    else:
        selected_data = get_filtered_retrieval_keyword_results(
            args.raw_keyword_urls_path
        )

    urls = [d["raw url"] for d in selected_data]
    images = [d["image urls"] for d in selected_data]

    if args.scrape_with_trafilatura:
        # Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            print(images[u])
            output.append(extract_info_trafilatura(urls[u], images[u]))
            # Only store in json file every 5 evidence
            if u % 5 == 0:
                save_result(output, args.trafilatura_path)
                output = []
    raise
    # Save all results in a Pandas Dataframe
    evidence_trafilatura = load_json(args.trafilatura_path)
    evidence = evidence_trafilatura.fillna("").to_dict(orient="records")
    # dataset = (
    #     load_json("dataset/train.json")
    #     + load_json("dataset/val.json")
    #     + load_json("dataset/test.json")
    # )
    # evidence = (
    #     merge_data(evidence_trafilatura, selected_data, dataset)
    #     .fillna("")
    #     .to_dict(orient="records")
    # )
    # Save the list of dictionaries as a JSON file
    with open(args.json_path, "w") as file:
        json.dump(evidence, file, indent=4)
