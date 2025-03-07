from dotenv import load_dotenv
from google.cloud import vision
import os
from openai import OpenAI
from tqdm import tqdm
import time
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.llm_prompting import gpt4_vision_prompting, gpt4_keyword_prompting, gpt4_prompting
from utils import *
from dataset_collection.scrape_utils import *
import argparse

import requests
import imghdr


def keyword_search(query, serper_api_key, num_results=30):
    url = "https://google.serper.dev/search/images"

    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": query,
        "num": num_results  # Increase the number of results
    }

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

            # print(f"Title: {title}\nSource: {source}\nURL: {link}\nImage: {image}\nPosition: {position}\n")

            results.append(r)

            # print("-" * 50)

        return results
    else:
        print("Error:", response.status_code, response.text)
        return {}


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
    # parser.add_argument(
    #     "--raw_keyword_urls_path",
    #     type=str,
    #     default="dataset/retrieval_results/keyword_evidence_urls.json",
    #     help="The json file to store the raw keyword search results.",
    # )
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
    serper_api_key = os.getenv("SERPER_API_KEY")

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
            story = gpt4_vision_prompting(prompt, client, path, max_tokens=200)
            # Generate a suitable query for web search
            query_prompt = (
                "Please formulate the given story into a suitable web search query. /n Story: "
                + str(story) + 
                "/n Output: Return reformulated web search query."
            )
            keyword_query, _ = gpt4_prompting(query_prompt, client)

            # Getting keyword search results
            keyword_search_results = keyword_search(
                keyword_query, serper_api_key
            )

            for r in keyword_search_results:
                merged_dict = {
                    "image path": path,
                    "story": story,
                    "query": keyword_query,
                    **r,  # merges all key/value pairs from r
                }
                raw_keyword_results.append(
                   merged_dict
                )

            time.sleep(args.sleep)
        with open(args.json_path, "w", encoding="utf-8") as file:
            # Save raw results
            json.dump(raw_keyword_results, file, indent=4)

        # # Apply filtering to the URLs to remove content produced by FC organizations and content that is not scrapable
        # selected_data = get_filtered_retrieval_keyword_results(
        #     args.raw_keyword_urls_path
        # )

    # else:
    #     selected_data = get_filtered_retrieval_keyword_results(
    #         args.raw_keyword_urls_path
    #     )

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
                output.append({
                    "image path": image_paths[u],
                    "story": story[u],
                    "query": query[u],
                    **result
                })
            
            # Only store in json file every 5 evidence
            if u % 3 == 0:
                save_keyword_result(output, args.trafilatura_path)
                output = []

    # NOTE: After that we need to download keyword images into keyword_images folder -> download_keyword_search_images.py

    
    # Save all results in a Pandas Dataframe
    # evidence_trafilatura = load_json(args.trafilatura_path)

    # df = pd.DataFrame(evidence_trafilatura)
    # df = df.fillna("")
    # df.head()
    # evidence = df.to_dict(orient="records")

    # with open(args.json_path, "w") as file:
    #     json.dump(evidence, file, indent=4)
