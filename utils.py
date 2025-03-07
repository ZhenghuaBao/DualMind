import os
import json
import glob
import base64
import requests
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def load_json(file_path):
    """
    Load json file
    """
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def concatenate_entry(d):
    """
    For all keys in a dictionary, if a value is a list, concatenate it.
    """
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = ";".join(
                map(str, value)
            )  # Convert list to a string separated by ';'
    return d


def append_to_json(file_path, data):
    """
    Append a dict or a list of dicts to a JSON file.
    """
    try:
        if not os.path.exists(file_path):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(file_path, "w") as file:
                json.dump([], file)
        # Open the existing file
        with open(file_path, "r+") as file:
            file_data = json.load(file)
            if type(data) == list:
                for d in data:
                    if type(d) == dict:
                        file_data.append(concatenate_entry(d))
            else:
                file_data.append(concatenate_entry(data))
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")


def save_result(output, json_file_path):
    """
    Save output results to a JSON file.
    """
    try:
        if type(output) == str:
            user_data = json.loads(output)
            append_to_json(json_file_path, user_data)
        else:
            append_to_json(json_file_path, output)
    except json.JSONDecodeError:
        # The output was not well formatted
        pass

def save_keyword_result(output, json_file_path):
    """
    Save output results to a JSON file.
    """
    try:
        if type(output) == str:
            user_data = json.loads(output)
            append_to_json(json_file_path, user_data)
        else:
            append_to_json(json_file_path, output)
    except json.JSONDecodeError:
        # The output was not well formatted
        pass


def entry_exists(json_file_path, url):
    """
    Check if an entry for the given URL already exists in the JSON file.
    """
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)
            return any(
                entry.get("URL").split("/")[-1] == url.split("/")[-1].split(".")[0]
                for entry in data
            )
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file.")
        return False
    except FileNotFoundError:
        return False


def is_folder_empty(folder_path):
    """
    Check if the given folder is empty.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return not os.listdir(folder_path)


def get_corpus(directory, json_file_path, image_directory):
    """
    Process each text file in the given directory.
    """
    text_files = []
    corpus = []
    # Identify the text files
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            text_files.append(os.path.join(directory, file))
    # Process each text file
    for txt_file in text_files:
        txt_file_name = os.path.basename(txt_file)
        image_folder_name = txt_file_name[:-4]  # Remove '.txt'
        image_folder_path = os.path.join(image_directory, image_folder_name)

        if is_folder_empty(image_folder_path):
            continue

        if entry_exists(json_file_path, txt_file):
            continue
        with open(txt_file, "r", encoding="utf-8") as file:
            text = file.read()
            text = text.split("Image URLs")[0]
            corpus.append(text)
    return corpus


def encode_image(image_path):
    """
    Encode images in base64. Format required by GPT4-Vision.
    """
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"


def get_valid_images(folder_path):
    """
    Returns a list of valid image file paths, ignoring .DS_Store and hidden files.
    """
    valid_images = []
    for filename in os.listdir(folder_path):
        if filename.startswith("."):  # Skip hidden files (including .DS_Store)
            continue
        file_path = os.path.join(folder_path, filename)
        valid_images.append(file_path)
    return valid_images


def download_images(image_url, filename):
    save_directory = "dataset/keyword_images"

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Create a session with retries
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Retry up to 5 times
        backoff_factor=2,  # Wait time increases exponentially (1s, 2s, 4s...)
        status_forcelist=[500, 502, 503, 504],  # Retry only on these errors
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Save the image
    path = os.path.join(save_directory, filename)

    try:
        response = session.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an error if status code is not 200

        with open(path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {image_url}: {e}")

    time.sleep(random.uniform(2, 5))  # Random delay to avoid bloc


def classify_similarity(cosine_score):
    if 0.8 <= cosine_score <= 1.0:
        return "High"
    elif 0.5 <= cosine_score < 0.8:
        return "Medium"
    else:
        return "Low"
