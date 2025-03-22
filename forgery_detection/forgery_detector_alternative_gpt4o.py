import os
from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64
import json

class ForgeryDetector:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.SYSTEM_PROMPT = (
            "You are a rigorous and responsible image tampering (altering) detection expert.  "
            "You can localize the exact tampered region and analyze your detection decision according to tampering clues at different levels.  "
            "Assuming that you have detected this is a Fake image and the manipulation type is one of {photoshop, ai-generated},  "
            "The following analysis supports the authenticity of the image, with observations categorized into high-level semantic coherence, middle-level visual consistency, and low-level pixel statistics. "
            "# High-Level Semantic Coherence "
            "Consistency with Common Sense "
            "The content is entirely plausible, aligning with real-world expectations. The scene reflects a natural, truthful setting with no misleading elements. "
            "# Middle-Level Visual Consistency "
            "Consistent Lighting "
            "The lighting throughout the image is coherent, with shadows, highlights, and reflections properly aligned with the light source, creating a realistic appearance. "
            "Compliance with Physical Constraints "
            "All interactions and placements of objects adhere to physical laws, such as gravity and balance, ensuring that the scene is plausible in the real world. "
            "Consistent Perspective "
            "The spatial relationships between objects are logically arranged, with no distortion, and the size, scale, and orientation of elements align with natural perspective rules. "
            "# Low-Level Pixel Statistics "
            "Cohesive Color Distribution "
            "The colors and tones in the image are smooth and cohesive, with no abrupt transitions, ensuring a natural blend across the scene. "
            "Uniform Texture and Sharpness "
            "The texture and sharpness are evenly distributed, with no areas appearing artificially smoothed, grainy, or oversharpened. "
            "Consistent Noise Patterns "
            "The noise distribution across the image is uniform, with no localized discrepancies or abrupt changes that could suggest editing or manipulation. Plese give your answser base on the image:"
        )

    def encode_image(self, image, quality=100):
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality) 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def forgery_image_analysis(self, image_path, quality=50):
        with Image.open(image_path) as img:
            img_b64_str = self.encode_image(img, quality=quality)
        img_type = "image/jpeg"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def forgery_image_detect(self, image_path, analysis, quality=50):
        with Image.open(image_path) as img:
            img_b64_str = self.encode_image(img, quality=quality)
        img_type = "image/jpeg"
        prompt_predict = (
            "You are an image forgery detector. "
            "Based on the semantic analysis provided for the image and the original image itself, "
            "determine if the image has been tampered with. You should analyze the image at three levels: "
            "high-level semantic coherence, middle-level visual consistency, and low-level pixel statistics. "
            "Based on these analyses, provide your result in the following JSON format:"
            "```json{"
            "\"forgery\": true/false,"
            "\"explain\": \"Explanation of your analysis at high-level semantic coherence, middle-level visual consistency, and low-level pixel statistics.\","
            "\"confidence\": 0-100"
            "}```"
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_predict + analysis},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )
        result = response.choices[0].message.content
        cleaned_result = result.strip('```json\n').strip('\n```')  # Remove the markdown block
        try:
            result_json = json.loads(cleaned_result)
        except json.JSONDecodeError:
            print(cleaned_result)
            return {"error": "Invalid response format from OpenAI API."}
        return result_json

def main():
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure you have your OpenAI API key set in the environment
    detector = ForgeryDetector(api_key)

    # Perform image forgery analysis
    image_path = "./dataset/mudi_test.png"
    analysis = detector.forgery_image_analysis(image_path)

    # Perform forgery detection
    detection_result = detector.forgery_image_detect(image_path, analysis)
    print(detection_result)

if __name__ == "__main__":
    main()
