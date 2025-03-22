import os
from PIL import Image


def find_invalid_images(image_dir):
    invalid_images = []

    # Iterate through all files in the directory
    for idx, filename in enumerate(os.listdir(image_dir)):
        file_path = os.path.join(image_dir, filename)
        # Check if the file is an image
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify if the image is valid
            except Exception as e:
                invalid_images.append((idx, filename, str(e)))

    return invalid_images


# NOTE: This function can be used to validate downloaded images
if __name__ == "__main__":
    image_directory = ""

    if not os.path.isdir(image_directory):
        print("Invalid directory path.")
    else:
        invalid_files = find_invalid_images(image_directory)

        if invalid_files:
            print("Invalid images found:")
            for idx, img, error in invalid_files:
                print(f"{idx},{img} - Error: {error}")
        else:
            print("All images are valid.")
