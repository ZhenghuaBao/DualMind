from PIL import Image
from baseline.generation_utils import *
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *


#############
#   GPT4   #
#############


def gpt4_prompting(content, client, max_tokens=1000):
    """
    Prompting the standard GPT4 model. Used for data labeling.
    """
    deployment_name = "gpt-4o-mini"
    messages = [{"role": "user", "content": content}]
    completion = client.chat.completions.create(
        model=deployment_name, messages=messages, max_tokens=max_tokens
    )
    output = completion.choices[0].message.content
    usage = completion.usage.total_tokens
    return output, usage


def gpt4_keyword_prompting(system_prompt, content, client, max_tokens=1000):
    """
    Prompting the standard GPT4 model. Used for data labeling.
    """
    deployment_name = "gpt-4o-mini"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    completion = client.chat.completions.create(
        model=deployment_name, messages=messages, max_tokens=max_tokens
    )
    output = completion.choices[0].message.content
    usage = completion.usage.total_tokens
    return output, usage


def gpt4_vision_prompting(
    prompt,
    client,
    image_path,
    map_manipulated_original={},
    modality="multimodal",
    temperature=0.2,
    max_tokens=50,
):
    """
    Prompting GPT4 multimodal.
    """
    deployment_name = "gpt-4o-mini"
    content = [{"type": "text", "text": prompt}]
    if modality in ["vision", "multimodal"]:
        if image_path in map_manipulated_original.keys():
            # Convert to the original image if it is detected as manipulated and an original is available
            image_path = map_manipulated_original[image_path]

        image64 = encode_image(image_path)
        content += [{"type": "image_url", "image_url": {"url": image64}}]
    messages = [{"role": "user", "content": content}]
    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
        output = completion.choices[0].message.content
    except Exception as e:
        output = e
    return output


# Special prompting for integrating forgery detection into prompt
def gpt4_vision_prompting_with_forgery_detection_(
    prompt,
    client,
    image_path,
    map_manipulated_original={},
    modality="vision",
    temperature=0.2,
    max_tokens=50,
    forgery_detection_explanation=None,
    forgery_detection_path=None,
):
    """
    Prompting GPT4 multimodal.
    """
    # deployment_name = 'gpt-4-turbo'
    deployment_name = "gpt-4o-mini"
    content = [{"type": "text", "text": prompt}]

    if modality in ["vision", "multimodal"]:
        if image_path in map_manipulated_original:
            # Convert to the original image (if the image is detected as manipulated and the original image is available)
            image_path = map_manipulated_original[image_path]

        try:
            with open(image_path, "rb") as image_file:
                image64 = (
                    f"data:image/{image_path.split('.')[-1]};base64,"
                    + base64.b64encode(image_file.read()).decode("utf-8")
                )
                content.append({"type": "image_url", "image_url": {"url": image64}})
        except FileNotFoundError as e:
            print(f"Error: Unable to locate the image file - {e}")
            output = "Error: Image file not found."
            return output
    messages = [{"role": "user", "content": content}]

    # If forgery detection information exists, process it as a separate prompt
    forgery_detection_content = (
        "Forgery detector did not find any forgery part in this image."
    )
    if forgery_detection_explanation and forgery_detection_path:
        try:
            with open(forgery_detection_path, "rb") as forgery_image_file:
                forgery_image64 = (
                    f"data:image/{forgery_detection_path.split('.')[-1]};base64,"
                    + base64.b64encode(forgery_image_file.read()).decode("utf-8")
                )
                forgery_detection_content = (
                    "This message is the output of the forgery detection. The image represents semantic segmentation of the "
                    "tampered area. Please be aware that forgery detection has a high false positive rate. Ignore any areas "
                    "that are detected as tampered if they are irrelevant to the claim stated below, even if the image shows "
                    "forgery in those regions. Below is the explanation for the detection:\n\n"
                    f"Forgery detection explanation: {forgery_detection_explanation}\n"
                    f"Forgery detection image (base64 encoded): {forgery_image64}"
                )
        except FileNotFoundError as e:
            pass
            # print(f"Error: Unable to locate the forgery detection image file - {e}")
            # forgery_detection_content = "Forgery detector did not find any forgery part in this image."

    # Add forgery detection information to content
    forgery_detection_content_message = [
        {"role": "user", "content": forgery_detection_content}
    ]
    messages.extend(forgery_detection_content_message)

    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
        output = completion.choices[0].message.content
        print(output)
    except Exception as e:
        print(f"Exception: {e}")
        output = "Error: Unable to complete the request."

    return output


def gpt4_vision_prompting_with_forgery_detection(
    prompt,
    client,
    image_path,
    map_manipulated_original={},
    modality="vision",
    temperature=0.2,
    max_tokens=50,
    forgery_detection_explanation=None,
    forgery_detection_path=None,
):
    """
    Prompting GPT4 multimodal.
    """
    # deployment_name = 'gpt-4-turbo'
    deployment_name = "gpt-4-turbo"
    content = [{"type": "text", "text": prompt}]

    # Process the image
    if modality in ["vision", "multimodal"]:
        if image_path in map_manipulated_original:
            # Convert to the original image (if the image is detected as manipulated and the original image is available)
            image_path = map_manipulated_original[image_path]

        try:
            with open(image_path, "rb") as image_file:
                image64 = (
                    f"data:image/{image_path.split('.')[-1]};base64,"
                    + base64.b64encode(image_file.read()).decode("utf-8")
                )
                content.append({"type": "image_url", "image_url": {"url": image64}})
        except FileNotFoundError as e:
            print(f"Error: Unable to locate the image file - {e}")
            output = "Error: Image file not found."
            return output

    # Process forgery detection content and add forgery detection image separately to content
    if forgery_detection_explanation and forgery_detection_path:
        try:
            with open(forgery_detection_path, "rb") as forgery_image_file:
                forgery_image64 = (
                    f"data:image/{forgery_detection_path.split('.')[-1]};base64,"
                    + base64.b64encode(forgery_image_file.read()).decode("utf-8")
                )

                # Add forgery detection image separately as image_url
                forgery_detection_content = (
                    "This message is the output of the forgery detection. The image represents semantic segmentation of the "
                    "tampered area. Please be aware that forgery detection has a high false positive rate. Ignore any areas "
                    "that are detected as tampered if they are irrelevant to the claim stated below, even if the image shows "
                    "forgery in those regions. Below is the explanation for the detection:\n\n"
                    f"Forgery detection explanation: {forgery_detection_explanation}"
                )

                # Add forgery detection explanation to content
                content.append({"type": "text", "text": forgery_detection_content})
                content.append(
                    {"type": "image_url", "image_url": {"url": forgery_image64}}
                )

        except FileNotFoundError as e:
            pass
            # If forgery detection image is not found, retain the default prompt
            forgery_detection_content = (
                "Forgery detector did not find any forgery part in this image."
            )
            content.append({"type": "text", "text": forgery_detection_content})

    # Create messages
    messages = [{"role": "user", "content": content}]

    try:
        completion = client.chat.completions.create(
            model=deployment_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
        # Get the model's output
        output = completion.choices[0].message.content
        print(output)
    except Exception as e:
        print(f"Exception: {e}")
        output = "Error: Unable to complete the request."

    return output


def assemble_prompt_gpt4(
    question, answer=None, evidence=[], demonstrations=[], modality="multimodal"
):
    """
    Assemble the prompt for GPT4.
    """
    # demonstrations are tuples (image_paths,answer,evidence_df)
    prompt = ""
    for d in range(len(demonstrations)):
        prompt += assemble_prompt_gpt4(
            question,
            answer=demonstrations[d][1],
            evidence=demonstrations[d][2],
            demonstrations=[],
            modality="evidence",
        )  # Demonstrations are provided without images
        prompt += "\n\n"

    if modality == "evidence":
        prompt += "You are given online articles that used a certain image. Your task is to answer a question about the image.\n\n"
    elif modality == "story":
        prompt += "You are given online articles that are related to the context of an image. Your task is to answer a question about the image.\n\n"
    elif modality == "multimodal":
        prompt += "You are given an image and online articles that used that image. Your task is to answer a question about the image using the image and the articles.\n\n"
    else:
        prompt += "You are given an image. Your task is to answer a question about the image.\n\n"
    if len(evidence) != 0:
        prompt += get_evidence_prompt(evidence)
    prompt += "Question: " + question + "\n"
    prompt += "Answer: "
    if answer:
        prompt += answer + "\n"
    return prompt


#############
#   Llava   #
#############


def llava_prompting(
    prompt,
    image_path,
    pipe,
    map_manipulated_original={},
    temperature=0.2,
    max_tokens=50,
):
    """
    Prompting Llava model for vision and multimodal input.
    """
    try:
        if image_path in map_manipulated_original.keys():
            # Convert to the original image if it is detected as manipulated and an original is available
            image_path = map_manipulated_original[image_path]
        image = Image.open(image_path)
        outputs = pipe(
            image,
            prompt=prompt,
            generate_kwargs={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
            },
        )
        return outputs[0]["generated_text"].split("ASSISTANT:")[1].split("\n\n")[0]
    except RuntimeError as e:
        print(e)
        return ""


def assemble_prompt_llava(
    question,
    answer=None,
    evidence=[],
    demonstrations=[],
    modality="vision",
):
    """
    Assemble the prompt for Llava.
    """
    prompt = ""
    # Demonstrations are tuples (image_paths,answer,evidence_df)
    for d in range(len(demonstrations)):
        prompt += assemble_prompt_llava(
            question,
            answer=demonstrations[d][1],
            evidence=demonstrations[d][2],
            demonstrations=[],
            modality="evidence",
        )
        prompt += "\n\n"
    prompt += "USER:"
    if modality == "evidence":
        prompt += "You are given online articles that used a certain image. Your task is to answer a question about the image.\n\n"
    elif modality == "multimodal":
        prompt += "You are given an image and online articles that used that image. Your task is to answer a question about the image using the image and the articles.\n\n"
        prompt += "<image>"
    else:
        prompt += "You are given an image. Your task is to answer a question about the image.\n\n"
        prompt += "<image>"
    if len(evidence) != 0:
        prompt += get_evidence_prompt(evidence)
    prompt += "Question: " + question + "\n"
    if answer:
        # Provide the answer for demonstrations
        prompt += "Answer:"
        prompt += answer + "\n"
    else:
        prompt += "ASSISTANT:"
    return prompt


#############
#   Llama   #
#############


def llama_prompting(prompt, pipeline, tokenizer, temperature, max_tokens):
    """
    Prompting Llama2 model for text input.
    """
    output = pipeline(
        prompt,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_tokens,
        temperature=temperature,
        do_sample=True,
    )["generated_text"]
    return output


def assemble_prompt_llama(
    question, answer=None, evidence=[], demonstrations=[], modality="evidence"
):
    """
    Assemble the prompt for Llama2.
    """
    prompt = "<s>[INST] <<SYS>>"
    prompt += "You are given online articles that used a certain image. Your task is to answer a question about the image.<</SYS>>"
    if len(evidence) != 0:
        prompt += get_evidence_prompt(evidence)
    prompt += "Question: " + question + "\n"
    prompt += "[/INST]"
    return prompt
