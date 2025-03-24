from google.cloud import vision
import os
from tqdm import tqdm
import time
import sys
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *
from dataset_collection.scrape_utils import *
import argparse
from urllib.parse import urlparse
import json
import pytineye

key_file_path = Path("dataset/key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path.as_posix()


def detect_web(path, how_many_queries=30):
    """
    Detects web annotations given an image.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection

    page_urls = []
    matching_image_urls = {}
    visual_entities = {}

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            page_urls.append(page.url)
            if page.full_matching_images:
                # List of image URLs for that webpage (the image can appear more than once)
                matching_image_urls[page.url] = [
                    image.url for image in page.full_matching_images
                ]
            else:
                matching_image_urls[page.url] = []
            if page.partial_matching_images:
                matching_image_urls[page.url] += [
                    image.url for image in page.partial_matching_images
                ]
    else:
        print("No matching images found for " + path)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            # Collect web entities as entity-score dictionary pairs
            visual_entities[entity.description] = entity.score

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return page_urls, matching_image_urls, visual_entities


def detect_tineye(path, max_results=30):
    # API URL, APIkey
    TINEYE_API_URL = "https://api.tineye.com/rest/"
    TINEYE_API_KEY = ""

    # Initialize the TinEye API request object
    api = pytineye.TinEyeAPIRequest(api_url=TINEYE_API_URL, api_key=TINEYE_API_KEY)

    page_urls = []
    matching_image_urls = {}

    # Check if it is an image file
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        with open(path, "rb") as img_file:
            image_data = img_file.read()

        # Reverse search with search_data
        try:
            response = api.search_data(image_data, limit=max_results)

            # Access matches directly via response.matches
            if len(response.matches) > 0:  # matches is a list
                for result in response.matches:  # Access to each match
                    page_url = result.image_url  # Get Image URL
                    if page_url not in page_urls:
                        page_urls.append(page_url)

                    if page_url not in matching_image_urls:
                        matching_image_urls[page_url] = []
                    matching_image_urls[page_url].append(
                        result.image_url
                    )  # Get the URL of the matching image

            else:
                print(f"No matches found for {path}.")
        except Exception as e:
            print(f"error occurs: {e}")
    else:
        print(f"Skip non-image files: {path}")

    return page_urls, matching_image_urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect evidence using Google Reverse Image Search."
    )
    parser.add_argument(
        "--collect_google",
        type=int,
        default=0,
        help="Whether to collect evidence URLs with the google API. If 0, it is assumed that a file containing URLs already exists.",
    )
    parser.add_argument(
        "--evidence_urls",
        type=str,
        default="dataset/retrieval_results/evidence_urls.json",
        help="Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.",
    )
    parser.add_argument(
        "--google_vision_api_key",
        type=str,
        default="",  # Provide your own key here as default value
        help="Your key to access the Google Vision services, including the web detection API. Only needed if collect_google is set to 1.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="dataset/processed_img/",
        help="The folder where the images are stored.",
    )
    parser.add_argument(
        "--raw_ris_urls_path",
        type=str,
        default="dataset/retrieval_results/ris_results.json",
        help="The json file to store the raw RIS results.",
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
        default="dataset/retrieval_results/trafilatura_data.json",
        help="The json file to store the scraped trafilatura  content as a json file.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="dataset/retrieval_results/evidence.json",
        help="The json file to store the text evidence as a json file.",
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
    parser.add_argument(
        "--collect_tineye",
        type=int,
        default=1,
        help="Whether to collect evidence URLs with TinEye API. If 0, assumes a file with URLs already exists.",
    )
    parser.add_argument(
        "--tineye_api_key",
        type=str,
        default=" ",  # Provide your TinEye API key
        help="Your key to access the TinEye API. Required if collect_tineye is set to 1.",
    )

    args = parser.parse_args()
    key = os.getenv(args.google_vision_api_key)

    # Create directories if they do not exist yet
    if not "retrieval_results" in os.listdir("dataset/"):
        os.mkdir("dataset/retrieval_results/")

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

    all_filtered_results = []  # For storing RIS results from Google + TinEye
    # Google RIS
    if args.collect_google:
        raw_ris_results = []
        filtered_results = []  # Storing filtered results

        for path in tqdm(os.listdir(args.image_path)):
            urls, image_urls, vis_entities = detect_web(
                args.image_path + path, args.max_results
            )

            for url in urls:
                credibility = get_credibility(
                    url
                )  # Calculate the reliability of a single URL

                # If the URL credibility is “Unknown, low”, skip it
                if credibility in ["Unknown", "Low"]:
                    continue

                url_image_urls = {}  # Stores the image link corresponding to the URL
                if url in image_urls:
                    url_image_urls = image_urls[
                        url
                    ]  # Directly get all the image links associated with the URL

                # Stores the results, with each URL as a separate entry
                filtered_results.append(
                    {
                        "image path": args.image_path + path,
                        "url": url,
                        "credibility": credibility,
                        "image urls": url_image_urls,  # Separate storage of images and credibility associated with each URL
                        "visual entities": vis_entities,  # Visual entity data is still associated with pictures
                    }
                )

                time.sleep(args.sleep)
        all_filtered_results.extend(
            filtered_results
        )  # Results pending merger with TinEye

    # Tineye RIS
    if args.collect_tineye:
        raw_ris_results = []
        filtered_results = []

        for path in tqdm(os.listdir(args.image_path)):
            urls, image_urls = detect_tineye(args.image_path + path, 10)

            for url in urls:
                credibility = get_credibility(url)

                if credibility in ["Unknown", "Low"]:
                    continue

                url_image_urls = {}

                if url in image_urls:
                    url_image_urls = image_urls[url]

                filtered_results.append(
                    {
                        "image path": args.image_path + path,
                        "url": url,
                        "credibility": credibility,
                        "image urls": url_image_urls,
                    }
                )

                time.sleep(args.sleep)
        all_filtered_results.extend(
            filtered_results
        )  #  Merging TinEye and google results.

        #  Save filtered results
    with open(args.raw_ris_urls_path, "w") as file:
        json.dump(all_filtered_results, file, indent=4)
        # Further filtering of URLs to remove unsuitable content for crawling
    selected_data = get_filtered_retrieval_results(args.raw_ris_urls_path)

    urls = [d["raw url"] for d in selected_data]
    images = [d["image urls"] for d in selected_data]

    if args.scrape_with_trafilatura:
        # Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            output.append(extract_info_trafilatura(urls[u], images[u]))
            # Only store in json file every 50 evidence
            if u % 1 == 0:
                save_result(output, args.trafilatura_path)
                output = []

    # Save all results in a Pandas Dataframe
    evidence_trafilatura = load_json(args.trafilatura_path)

    dataset = (
        load_json("dataset/train.json")
        + load_json("dataset/val.json")
        + load_json("dataset/test.json")
    )

    evidence = (
        merge_data(evidence_trafilatura, selected_data, dataset)
        .fillna("")
        .to_dict(orient="records")
    )
    # print(f"Number of successful Trafilatura resolutions: {len(evidence_trafilatura)}")

    # Save the list of dictionaries as a JSON file
    with open(args.json_path, "w") as file:
        json.dump(evidence, file, indent=4)
